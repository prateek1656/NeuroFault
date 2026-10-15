"""Regression tests for three real bugs found and fixed while planning this
suite - each verified to reproduce against the pre-fix code before being
fixed (verified directly), not written to pass against already-correct code.
"""

from __future__ import annotations

import pytest
import torch

from neurofault.config import CrossbarConfig, FaultConfig, MitigationConfig
from neurofault.crossbar.health_monitor import HealthMonitor
from neurofault.crossbar.self_healing import build_redundancy_pool
from neurofault.faults.injection import inject_faults
from neurofault.mitigation.dispatch import apply_runtime_mitigation
from tests.unit.fakes import make_handle


def test_stuck_cell_health_never_recovers_across_many_health_checks():
    """Bug: update_health_metrics() used to recompute stress/health_scores
    for every cell including stuck ones, whose stress decayed toward a small
    steady state every call regardless of being permanently dead - a cell
    injected with health_scores=0 reached 74.73 after 200 calls."""
    handle = make_handle(shape=(20, 16))
    monitor = HealthMonitor(handle)
    fault_config = FaultConfig(density=0.3)

    inject_faults(handle, monitor, fault_config)
    stuck = monitor.stuck_mask.clone()
    assert stuck.any()

    health_before = monitor.health_scores[stuck].clone()
    stress_before = monitor.stress[stuck].clone()
    stability_before = monitor.stability_index[stuck].clone()

    for _ in range(200):
        monitor.update_health_metrics()

    assert torch.equal(monitor.health_scores[stuck], health_before)
    assert torch.equal(monitor.stress[stuck], stress_before)
    assert torch.equal(monitor.stability_index[stuck], stability_before)


def test_non_stuck_cell_health_still_updates_normally():
    """The fix must not freeze the whole monitor - only stuck positions."""
    handle = make_handle(shape=(20, 16))
    monitor = HealthMonitor(handle)
    fault_config = FaultConfig(density=0.3)

    inject_faults(handle, monitor, fault_config)
    non_stuck = ~monitor.stuck_mask
    stress_before = monitor.stress[non_stuck].clone()

    voltage = torch.ones(20, 16) * 0.5
    for _ in range(50):
        monitor.update_health_metrics(voltage_applied=voltage, write_op=True)

    assert not torch.equal(monitor.stress[non_stuck], stress_before)


def test_soft_mitigation_never_touches_stuck_cells():
    """Bug: dispatch.py passed the raw, unfiltered critical_devices list
    (which includes stuck cells - they trivially satisfy the
    extreme-conductance check) into apply_soft_mitigation() - of 73 critical
    devices at density 0.2, 64 were stuck and all 64 got their conductance
    changed by a mechanism that models drift correction, not un-sticking a
    hardware-stuck-at device (see mitigation/soft.py's docstring)."""
    handle = make_handle(shape=(20, 16))
    monitor = HealthMonitor(handle)
    pool = build_redundancy_pool(handle, CrossbarConfig())
    fault_config = FaultConfig(density=0.2)

    inject_faults(handle, monitor, fault_config)
    stuck = monitor.stuck_mask.clone()
    matrix_before = handle.accessor.read().clone()

    mitigation_config = MitigationConfig(
        enable_soft_mitigation=True,
        enable_remapping=False,
        enable_layer_reset=False,
        health_threshold=1.0,  # low enough that avg_health > threshold, taking the soft path
    )
    monitor.update_health_metrics()
    critical = monitor.get_critical_devices()
    assert any(stuck[r, c] for r, c in critical)  # precondition: stuck cells ARE flagged critical

    apply_runtime_mitigation(
        handle,
        pool,
        monitor,
        critical,
        monitor.get_health_summary()["avg_health"],
        mitigation_config,
    )

    matrix_after = handle.accessor.read()
    assert torch.equal(matrix_before[stuck], matrix_after[stuck])


def test_soft_mitigation_only_touches_verified_drift_cells():
    """Real precision fix found during development, superseding this test's original assertion: filtering on
    "not stuck" still let through cells that are merely legitimately-trained
    near an extreme conductance value - get_critical_devices()'s heuristic
    can't distinguish that from real drift, so soft_mitigation would
    "correct" (damage) perfectly good learned weights. It must now touch
    only monitor.drift_mask-confirmed cells - verified here for both a
    critical-but-unverified cell (left alone) and a genuinely drift-marked
    one (nudged), in the same crossbar."""
    handle = make_handle(shape=(20, 16))
    monitor = HealthMonitor(handle)
    pool = build_redundancy_pool(handle, CrossbarConfig())

    # (0, 0): critical via a raw extreme conductance write, but never
    # actually drifted by the fault injector - e.g. a legitimately-trained
    # weight that happens to need an extreme value. Must be left alone.
    # (1, 1): genuinely drift-affected (failure_model="drift" would set
    # this) - must be nudged.
    matrix = handle.accessor.read()
    matrix[0, 0] = monitor.g_max
    matrix[1, 1] = monitor.g_max
    handle.accessor.write(matrix)
    monitor.drift_mask[1, 1] = True

    mitigation_config = MitigationConfig(
        enable_soft_mitigation=True,
        enable_remapping=False,
        enable_layer_reset=False,
        health_threshold=1.0,
    )
    monitor.update_health_metrics()
    critical = monitor.get_critical_devices()
    assert (0, 0) in critical
    assert (1, 1) in critical
    assert not monitor.stuck_mask[0, 0]
    assert not monitor.stuck_mask[1, 1]

    outcome = apply_runtime_mitigation(
        handle,
        pool,
        monitor,
        critical,
        monitor.get_health_summary()["avg_health"],
        mitigation_config,
    )

    assert outcome.soft_mitigated == 1
    assert (1, 1) in outcome.soft_mitigated_positions
    matrix_after = handle.accessor.read()
    assert matrix_after[0, 0] == monitor.g_max  # unverified cell left alone
    assert matrix_after[1, 1] != monitor.g_max  # verified drift cell nudged toward g_mid


def test_remapping_only_touches_verified_stuck_cells():
    """Real precision fix found during development, the same class of issue as soft_mitigation above: remapping
    used to act on the raw, unfiltered get_critical_devices() heuristic set,
    which can include cells that are merely legitimately-trained near an
    extreme value, not actually faulty. Remapping is a permanent, capacity-
    limited hardware action (RedundancyPool.remapping) and should only ever
    be spent on a verified-dead cell (stuck_mask), never a heuristically-
    flagged one."""
    handle = make_handle(shape=(20, 16))
    monitor = HealthMonitor(handle)
    pool = build_redundancy_pool(handle, CrossbarConfig())

    # (0, 0): critical via a raw extreme conductance write, never actually
    # stuck. (1, 1): genuinely stuck (what inject_faults() would produce).
    matrix = handle.accessor.read()
    matrix[0, 0] = monitor.g_max
    matrix[1, 1] = monitor.g_max
    handle.accessor.write(matrix)
    monitor.stuck_mask[1, 1] = True

    mitigation_config = MitigationConfig(
        enable_soft_mitigation=False,
        enable_remapping=True,
        enable_layer_reset=False,
        health_threshold=1.0,
    )
    monitor.update_health_metrics()
    critical = monitor.get_critical_devices()
    assert (0, 0) in critical
    assert (1, 1) in critical

    outcome = apply_runtime_mitigation(
        handle,
        pool,
        monitor,
        critical,
        monitor.get_health_summary()["avg_health"],
        mitigation_config,
    )

    assert outcome.remapped == 1
    matrix_after = handle.accessor.read()
    assert matrix_after[0, 0] == monitor.g_max  # unverified cell left alone
    assert matrix_after[1, 1] != monitor.g_max  # verified stuck cell remapped


def test_unknown_redundancy_scheme_raises():
    """Bug: an unrecognized redundancy_scheme string (a typo included)
    silently fell back to idealized_per_cell - verified directly with
    redundancy_scheme='row_col_granularityy' (one extra letter). Now fails
    this fast at CrossbarConfig construction; build_redundancy_pool()'s own
    check (still exercised directly here) stays as defense-in-depth for a
    config mutated after construction, which this codebase does elsewhere
    (dataclasses don't re-run __post_init__ on attribute assignment)."""
    with pytest.raises(ValueError, match="Unknown crossbar_config.redundancy_scheme"):
        CrossbarConfig(redundancy_scheme="row_col_granularityy")

    handle = make_handle(shape=(8, 8))
    config = CrossbarConfig()
    config.redundancy_scheme = "row_col_granularityy"  # bypasses __post_init__
    with pytest.raises(ValueError, match="Unknown crossbar_config.redundancy_scheme"):
        build_redundancy_pool(handle, config)


def test_unknown_detection_method_raises():
    """Bug: an unrecognized detection_method silently behaved as
    'full_diff' - the `if detection_method == "checksum"` check simply never
    triggered, no error, no warning."""
    handle = make_handle(shape=(8, 8))
    monitor = HealthMonitor(handle)

    with pytest.raises(ValueError, match="Unknown detection_method"):
        monitor.get_critical_devices(detection_method="Checksum")  # wrong case
