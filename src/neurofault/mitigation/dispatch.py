"""Central dispatch for which mitigation strategy runs on a set of critical
devices.

Fixes the original codebase's bug #2: `_apply_runtime_fault_mitigation`'s
final `else` branch called `reset_critical_layer` UNCONDITIONALLY, with no
`enable_layer_reset` check - meaning a "No Mitigations" or "Soft Mitigation
Only" experiment config could still silently trigger a full layer reset.

Every branch here is guarded by its own `mitigation_config.enable_*` flag
with no exception - if a branch's condition is met but its flag is off,
nothing happens at all, full stop. There is no branch that calls a
mitigation function unconditionally.

Also fixes bug #4: soft mitigation is reachable directly from this runtime
path (not only from a separate predictive path gated on history that may
never accumulate).
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field

from neurofault.config import MitigationConfig
from neurofault.crossbar.array import CrossbarHandle
from neurofault.crossbar.health_monitor import HealthMonitor
from neurofault.crossbar.self_healing import RedundancyPool, RowColumnRedundancyPool
from neurofault.mitigation.remap import apply_remap_with_positions
from neurofault.mitigation.soft import apply_soft_mitigation

logger = logging.getLogger(__name__)


@dataclass
class MitigationOutcome:
    soft_mitigated: int = 0
    remapped: int = 0
    layer_reset: bool = False
    actions_taken: list[str] = field(default_factory=list)
    # Exact (row, col) positions soft-mitigated this call - system.py needs these to
    # pin the correction through the next synchronize() (see its module docstring):
    # a conductance-only write here would otherwise be silently erased before any
    # forward pass ever benefits from it. Remap doesn't need an equivalent: its
    # corrected positions are already tracked permanently by the pool itself
    # (RedundancyPool.remapping / RowColumnRedundancyPool.condemned_rows/cols).
    soft_mitigated_positions: list[tuple[int, int]] = field(default_factory=list)
    # Exact per-cell positions remapped this call (empty for row_col_granularity,
    # whose unit is rows/columns). Already marked in monitor.remapped_mask.
    remapped_positions: list[tuple[int, int]] = field(default_factory=list)


RemapSelector = Callable[[list[tuple[int, int]]], list[tuple[int, int]]]


def apply_stuck_remap(
    handle: CrossbarHandle,
    pool: RedundancyPool | RowColumnRedundancyPool,
    monitor: HealthMonitor,
    candidates: list[tuple[int, int]],
    remap_selector: RemapSelector | None = None,
) -> tuple[int, list[tuple[int, int]]]:
    """Remap step shared by the periodic dispatch below and the policy-driven
    system.remap_now(): optionally order/truncate `candidates` (effective-stuck
    positions) with `remap_selector`, spend spares on the result, and tell the
    monitor. remap_selector=None is the identity - the legacy row-major prefix
    behavior, byte-for-byte. The selector sees positions only (it is a closure
    built by system.py, which owns the per-layer weight/fisher state it needs), so
    the pool's prefix-until-capacity rule stays untouched: selection POLICY
    (which cells deserve spares) lives upstream of the pool's selection MECHANISM."""
    if remap_selector is not None:
        candidates = remap_selector(candidates)
    remapped, positions = apply_remap_with_positions(handle, pool, candidates)
    if positions:
        rows_idx, cols_idx = zip(*positions)
        monitor.mark_remapped(list(rows_idx), list(cols_idx))
    return remapped, positions


def apply_runtime_mitigation(
    handle: CrossbarHandle,
    pool: RedundancyPool | RowColumnRedundancyPool,
    monitor: HealthMonitor,
    critical_devices: list[tuple[int, int]],
    avg_health: float,
    mitigation_config: MitigationConfig,
    reset_fn=None,  # Callable[[], bool] | None - injected so dispatch doesn't need
    #                 to know layer-reset's full signature; None means "not available"
    remap_selector: RemapSelector | None = None,  # ordering/budget for remap candidates; None =
    #                 identity (legacy prefix order)
) -> MitigationOutcome:
    outcome = MitigationOutcome()
    if not critical_devices:
        return outcome

    threshold = mitigation_config.health_threshold

    # Real precision fix found during development: get_critical_devices() flags cells via a broad
    # heuristic
    # (health score, failure probability, OR extreme conductance) that can
    # include cells that are simply legitimately-trained near an extreme
    # value, not actually faulty - remapping/layer_reset should only ever
    # touch cells confirmed permanently, physically dead (stuck_mask), never
    # a merely-heuristically-flagged one. This is the mechanistic explanation
    # during development found for why layer_reset kept hurting even once made
    # "targeted": it was still targeting the broad heuristic set, not
    # verified failures.
    # Vectorized (see HealthMonitor.filter_positions_by_mask's docstring for the real
    # perf bug this replaced - a Python loop with per-element tensor indexing that was
    # ~90% of check_health()'s cost once critical_devices reaches hundreds of thousands
    # of entries on ResNet-9's largest crossbars).
    # effective_stuck, not stuck_mask (2026-10-02, remap bug #1): cells already
    # remapped stay stuck_mask=True as the record of their original defect, but
    # are served by a working spare. Leaving them in this list made
    # RedundancyPool.select_remappable() stop at index 0 in every wave after the
    # first - verified from a real run.log ("Remapped 14745/14745 (7373
    # remaining)" in wave 1, "Remapped 0/14745" in wave 2), ~5% of every pool
    # never used.
    stuck_capable = monitor.filter_positions_by_mask(critical_devices, monitor.effective_stuck)

    # Remap is NOT gated on avg_health (2026-10-02, remap bug #2). It used to
    # run only when per-layer avg_health > health_threshold, but avg_health is
    # a synthetic stress-model quantity, not a measurement of the stuck-cell
    # repair problem: verified by probe it drifts 100 -> 90 with ZERO faults
    # purely from the number of health checks run, and in a real 3-wave run the
    # per-layer value was 85-86 / 75.1-76.3 / 67.7-68.6 at waves 1/2/3 - wave 2
    # passed the 75 gate by 0.1-0.3 by luck of check count and wave 3 could
    # never remap at all. Whether a verified-dead (stuck_mask) cell can be
    # bypassed depends only on spare capacity, so that is the only gate left
    # (inside the pool). soft_mitigation (drift) and layer_reset below keep
    # their health gating: their appropriateness is a statement about the
    # layer's overall condition, which is what that heuristic was built for.
    forced = mitigation_config.force_remapping and mitigation_config.enable_remapping
    if mitigation_config.enable_remapping and (stuck_capable or forced):
        if stuck_capable:
            outcome.remapped, outcome.remapped_positions = apply_stuck_remap(
                handle, pool, monitor, stuck_capable, remap_selector
            )
        outcome.actions_taken.append("forced_remap" if forced else "remap")
    if forced:
        return outcome

    if avg_health > threshold:
        if mitigation_config.enable_soft_mitigation:
            # Real bug found and fixed during development: soft mitigation models
            # correcting drift, not un-sticking a hardware-stuck-at device
            # (see soft.py's own docstring) - but critical_devices here is
            # unfiltered, and a stuck cell trivially satisfies
            # get_critical_devices()'s extreme-conductance check. Verified
            # directly: of 73 critical devices at fault density 0.2, 64 were
            # stuck_mask=True, and all 64 had their conductance changed by
            # "soft mitigation" - a physically impossible repair.
            #
            # Tightened further during development: filtering on "not stuck" still
            # let through cells that are merely legitimately-trained near an
            # extreme conductance value (get_critical_devices()'s heuristic
            # can't tell that apart from real drift) - soft_mitigation would
            # then "correct" (i.e. damage) perfectly good learned weights.
            # Filter on monitor.drift_mask instead - only cells the fault
            # injector has actually, verifiably drifted (failure_model=
            # "drift", see injection.py) are drift_capable now.
            drift_capable = monitor.filter_positions_by_mask(critical_devices, monitor.drift_mask)
            outcome.soft_mitigated = apply_soft_mitigation(handle, monitor, drift_capable)
            outcome.soft_mitigated_positions = drift_capable
            outcome.actions_taken.append("soft_mitigation")
        return outcome

    if avg_health > threshold / 2:
        if mitigation_config.enable_layer_reset and reset_fn is not None:
            outcome.layer_reset = reset_fn()
            outcome.actions_taken.append("layer_reset")
        # If layer reset is disabled here, nothing happens - no fallthrough.
        return outcome

    # Critical health: layer reset is the last resort, but ONLY if enabled.
    # The original codebase called reset_critical_layer here with no flag
    # check at all - that unconditional call is exactly what this dispatch
    # structure makes impossible.
    if mitigation_config.enable_layer_reset and reset_fn is not None:
        outcome.layer_reset = reset_fn()
        outcome.actions_taken.append("layer_reset")

    if not outcome.actions_taken:
        logger.debug(
            "%d critical devices in %s but no mitigation enabled for current health (%.1f%%)",
            len(critical_devices),
            handle.name,
            avg_health,
        )

    return outcome
