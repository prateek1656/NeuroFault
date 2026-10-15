"""Physical/state invariants that must hold regardless of how a crossbar got
into its current state - bounds, no NaN/Inf, no double-consumed redundancy
capacity. Property-based (hypothesis) where a sequence of random operations
is the natural way to fuzz these, plain pytest where a single deterministic
scenario already covers the invariant.
"""

from __future__ import annotations

import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from neurofault.config import CrossbarConfig, FaultConfig
from neurofault.crossbar.health_monitor import HealthMonitor
from neurofault.crossbar.self_healing import (
    RedundancyPool,
    RowColumnRedundancyPool,
    build_redundancy_pool,
)
from neurofault.faults.injection import inject_faults
from neurofault.mitigation.soft import apply_soft_mitigation
from tests.unit.fakes import make_handle


@given(
    voltages=st.lists(
        st.floats(min_value=-2.0, max_value=2.0, allow_nan=False), min_size=1, max_size=30
    ),
    write_ops=st.lists(st.booleans(), min_size=1, max_size=30),
)
@settings(max_examples=50, deadline=None)
def test_health_metrics_always_stay_within_documented_bounds(voltages, write_ops):
    """health_scores in [0,100], stability_index/stress in [0,1] - the
    torch.clamp calls are supposed to guarantee this everywhere, this fuzzes
    many operation sequences to actually verify it, not just the 1-2
    sequences today's other tests happen to exercise."""
    handle = make_handle(shape=(10, 10))
    monitor = HealthMonitor(handle)
    inject_faults(handle, monitor, FaultConfig(density=0.1))

    n = min(len(voltages), len(write_ops))
    for i in range(n):
        voltage = torch.full((10, 10), voltages[i])
        monitor.update_health_metrics(voltage_applied=voltage, write_op=write_ops[i])

        assert monitor.health_scores.min() >= 0.0
        assert monitor.health_scores.max() <= 100.0
        assert monitor.stability_index.min() >= 0.0
        assert monitor.stability_index.max() <= 1.0
        assert monitor.stress.min() >= 0.0
        assert monitor.stress.max() <= 1.0
        assert torch.isfinite(monitor.health_scores).all()
        assert torch.isfinite(monitor.stress).all()
        assert torch.isfinite(monitor.stability_index).all()


@given(density=st.floats(min_value=0.0, max_value=0.9))
@settings(max_examples=30, deadline=None)
def test_fault_injection_never_produces_nan_or_inf(density):
    handle = make_handle(shape=(16, 16))
    monitor = HealthMonitor(handle)
    inject_faults(handle, monitor, FaultConfig(density=density))

    matrix = handle.accessor.read()
    assert torch.isfinite(matrix).all()


def test_soft_mitigation_never_produces_nan_or_inf():
    handle = make_handle(shape=(16, 16))
    monitor = HealthMonitor(handle)
    inject_faults(handle, monitor, FaultConfig(density=0.3))
    monitor.update_health_metrics()
    critical = [(r, c) for r, c in monitor.get_critical_devices() if not monitor.stuck_mask[r, c]]

    apply_soft_mitigation(handle, monitor, critical)

    assert torch.isfinite(handle.accessor.read()).all()


@given(
    positions=st.lists(
        st.tuples(st.integers(min_value=0, max_value=15), st.integers(min_value=0, max_value=15)),
        min_size=1,
        max_size=100,
    )
)
@settings(max_examples=100, deadline=None)
def test_redundancy_pool_capacity_never_goes_negative_or_double_consumed(positions):
    handle = make_handle(shape=(16, 16))
    pool = RedundancyPool(handle, redundancy_factor=0.2)

    seen = set()
    for row, col in positions:
        ok = pool.remap_device(row, col)
        if (row, col) in seen:
            assert ok is False  # already-remapped position must never succeed twice
        seen.add((row, col))

        assert pool.available_capacity >= 0


@given(
    positions=st.lists(
        st.tuples(st.integers(min_value=0, max_value=15), st.integers(min_value=0, max_value=15)),
        min_size=0,
        max_size=100,
        unique=True,  # real critical_devices batches are always distinct - see remap.py
    )
)
@settings(max_examples=100, deadline=None)
def test_select_remappable_matches_calling_remap_device_in_a_loop(positions):
    """RedundancyPool.select_remappable (the vectorized batch form added
    2026-09-29 to fix the remap perf bug, see self_healing.py's docstring)
    must be exactly equivalent to the original call-remap_device-in-a-loop-
    and-break-on-first-False form it replaced, for any batch of distinct
    positions."""
    handle_a = make_handle(shape=(16, 16))
    pool_a = RedundancyPool(handle_a, redundancy_factor=0.2)
    expected = []
    for row, col in positions:
        if not pool_a.remap_device(row, col):
            break
        expected.append((row, col))

    handle_b = make_handle(shape=(16, 16))
    pool_b = RedundancyPool(handle_b, redundancy_factor=0.2)
    actual = pool_b.select_remappable(positions)

    assert actual == expected
    assert pool_a.remapping == pool_b.remapping


@given(
    critical_batches=st.lists(
        st.lists(
            st.tuples(
                st.integers(min_value=0, max_value=19), st.integers(min_value=0, max_value=19)
            ),
            min_size=0,
            max_size=20,
        ),
        min_size=1,
        max_size=10,
    )
)
@settings(max_examples=50, deadline=None)
def test_row_col_redundancy_pool_capacity_never_negative_or_double_condemned(critical_batches):
    handle = make_handle(shape=(20, 20))
    pool = RowColumnRedundancyPool(handle, redundancy_factor=0.3, condemn_threshold=0.1)

    for batch in critical_batches:
        new_rows, new_cols = pool.condemn_rows_and_cols(batch)

        assert pool.available_row_capacity >= 0
        assert pool.available_col_capacity >= 0
        assert len(set(new_rows)) == len(new_rows)  # no row appears twice in one return
        assert len(set(new_cols)) == len(new_cols)
        assert not (set(new_rows) & (pool.condemned_rows - set(new_rows)))  # no re-condemn


def test_reference_conductance_is_captured_before_any_fault_injection():
    """self_healing.py's own docstring assumes the pool is built before any
    fault is injected (the reference must be pristine) - this makes that
    ordering assumption an explicit, checked contract instead of an implicit
    one nothing verifies."""
    handle = make_handle(shape=(8, 8))
    pristine = handle.accessor.read().clone()

    pool = build_redundancy_pool(handle, CrossbarConfig())

    assert torch.equal(pool.reference_conductance, pristine)

    monitor = HealthMonitor(handle)
    inject_faults(handle, monitor, FaultConfig(density=0.5))

    # The pool's reference must NOT reflect the faults injected after it was built.
    assert torch.equal(pool.reference_conductance, pristine)
    assert not torch.equal(handle.accessor.read(), pristine)
