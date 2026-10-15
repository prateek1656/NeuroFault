"""Verify specific, stated claims in this codebase's own docstrings hold
generally, not just in the one example each existing test happens to use -
and verify the exact boundary behavior of branches whose off-by-one/wrong-
comparison-operator bugs would only show up exactly at a threshold value.
"""

from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

from neurofault.config import CrossbarConfig, MitigationConfig
from neurofault.crossbar.health_monitor import HealthMonitor
from neurofault.crossbar.self_healing import build_redundancy_pool
from neurofault.mitigation.dispatch import apply_runtime_mitigation
from tests.unit.fakes import make_handle


@given(
    row=st.integers(min_value=0, max_value=4),
    col=st.integers(min_value=0, max_value=4),
    size=st.integers(min_value=2, max_value=3),
)
@settings(max_examples=30, deadline=None)
def test_checksum_blind_spot_holds_for_many_canceling_patterns(row, col, size):
    """The single hardcoded 2x2 case in tests/unit/test_health_monitor.py,
    generalized: an NxN "conjugate" block (alternating g_max/g_min so every
    touched row's and column's sum contribution cancels) must always be
    invisible to checksum detection while full_diff always finds it -
    for many block sizes/positions, not just one."""
    shape = (8, 8)
    if row + size > shape[0] or col + size > shape[1]:
        return  # skip out-of-bounds placements; hypothesis will find in-bounds ones too

    handle = make_handle(shape=shape)
    monitor = HealthMonitor(handle)
    matrix = handle.accessor.read()

    block_cells = []
    for i in range(size):
        for j in range(size):
            # Checkerboard sign within the block - each row/col of the block
            # gets an equal count of g_max/g_min when size is even; only test
            # even sizes, where the cancellation is exact.
            if size % 2 != 0:
                return
            value = monitor.g_max if (i + j) % 2 == 0 else monitor.g_min
            matrix[row + i, col + j] = value
            block_cells.append((row + i, col + j))
    handle.accessor.write(matrix)

    via_checksum = monitor.get_critical_devices(detection_method="checksum")
    via_full_diff = set(monitor.get_critical_devices(detection_method="full_diff"))

    assert via_checksum == []
    assert set(block_cells) <= via_full_diff


def test_detection_cost_formula_is_exact_across_repeated_checks_with_and_without_faults():
    handle = make_handle(shape=(10, 5))
    monitor = HealthMonitor(handle)
    rows, cols = 10, 5

    # 3 clean checks: always the cheap rows+cols cost.
    for _ in range(3):
        monitor.get_critical_devices(detection_method="checksum")
    assert monitor.total_detection_cost == 3 * (rows + cols)

    # Inject a real fault - next checksum call must fall through to the full
    # rows*cols cost, exactly once, added on top of the prior total.
    matrix = handle.accessor.read()
    matrix[0, 0] = monitor.g_max
    handle.accessor.write(matrix)
    monitor.get_critical_devices(detection_method="checksum")

    assert monitor.total_detection_cost == 3 * (rows + cols) + (rows * cols)
    assert monitor.last_detection_cost == rows * cols


def _dispatch_at_health(avg_health: float, enable_soft: bool, enable_layer_reset: bool):
    handle = make_handle(shape=(8, 8))
    monitor = HealthMonitor(handle)
    # soft_mitigation now targets drift_mask specifically (dispatch.py's precision fix) - mark the
    # manufactured critical cell as drift-affected so tests here exercise
    # the health-threshold branching logic these tests are actually about,
    # not the (separately-tested) drift-filtering behavior.
    monitor.drift_mask[0, 0] = True
    pool = build_redundancy_pool(handle, CrossbarConfig())
    mitigation_config = MitigationConfig(
        enable_soft_mitigation=enable_soft,
        enable_remapping=False,
        enable_layer_reset=enable_layer_reset,
        health_threshold=50.0,
    )
    critical = [(0, 0)]
    reset_calls = []

    def reset_fn():
        reset_calls.append(True)
        return True

    outcome = apply_runtime_mitigation(
        handle, pool, monitor, critical, avg_health, mitigation_config, reset_fn=reset_fn
    )
    return outcome, reset_calls


def test_soft_mitigation_does_not_fire_at_exactly_the_health_threshold():
    """avg_health == threshold must NOT take the `> threshold` branch - a
    `>=` typo here would silently soft-mitigate one health point earlier
    than documented."""
    outcome, _ = _dispatch_at_health(50.0, enable_soft=True, enable_layer_reset=True)
    assert outcome.soft_mitigated == 0
    assert "soft_mitigation" not in outcome.actions_taken


def test_soft_mitigation_fires_just_above_the_health_threshold():
    outcome, _ = _dispatch_at_health(50.0001, enable_soft=True, enable_layer_reset=False)
    assert outcome.soft_mitigated > 0


def test_layer_reset_is_the_only_branch_at_and_below_half_threshold():
    """Both the `> threshold/2` and the final `else` branch call the exact
    same layer_reset action - soft mitigation/remapping must never fire in
    either, checked directly at the threshold/2 boundary itself."""
    outcome_at_half, reset_calls_at_half = _dispatch_at_health(
        25.0, enable_soft=True, enable_layer_reset=True
    )
    assert outcome_at_half.soft_mitigated == 0
    assert len(reset_calls_at_half) == 1

    outcome_below_half, reset_calls_below_half = _dispatch_at_health(
        10.0, enable_soft=True, enable_layer_reset=True
    )
    assert outcome_below_half.soft_mitigated == 0
    assert len(reset_calls_below_half) == 1


def test_no_mitigation_action_taken_when_nothing_is_enabled():
    """The dispatch structure's whole point (see dispatch.py's docstring):
    if a branch's condition is met but its flag is off, nothing happens -
    never an unconditional fallback action."""
    for avg_health in (60.0, 25.0, 5.0):
        outcome, reset_calls = _dispatch_at_health(
            avg_health, enable_soft=False, enable_layer_reset=False
        )
        assert outcome.actions_taken == []
        assert reset_calls == []
