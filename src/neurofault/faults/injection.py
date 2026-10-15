"""Fault injection for a single crossbar.

Four fault models, selected via FaultConfig.failure_model:

- "density" (default): vectorized custom stuck-at, a direct rewrite of the
  original SelfHealingCrossbar.inject_faults with its two Python for-loops
  replaced by tensor indexing - this is the path the Phase 1 smoke test
  actually exercises and must work. Memoryless and purely spatial: picks
  density * total_cells cells fresh each call via `distribution`.
- "weibull" / "exponential": real time-to-failure models (see
  faults/reliability.py) - each cell's own accumulated simulated cycles
  (HealthMonitor.cycles) determines its own conditional failure probability
  this interval, optionally accelerated by Arrhenius temperature scaling.
  Deliberately orthogonal to the spatial `distribution` option (the two are not
  combined here).
- "drift": a real, distinct failure mode found missing during development (see
  config.py's FaultConfig docstring) - gradual,
  cumulative conductance shift toward one rail, never marked stuck_mask
  (the cell stays functional, just imprecise) - the failure mode
  mitigation/soft.py's own docstring says it corrects, which this project
  never actually generated before now. Kept as an independently-selectable
  failure_model, not force-combined with stuck-at, so each mitigation
  strategy can first be evaluated against the fault type it targets.

All paths are fully vectorized - no Python per-cell loops, matching this
project's standing discipline (health_monitor.py's own docstring records a
real run killed by exactly that kind of loop).
"""

from __future__ import annotations

import logging

import torch

from neurofault.config import FaultConfig
from neurofault.crossbar.array import CrossbarHandle
from neurofault.crossbar.health_monitor import HealthMonitor
from neurofault.faults.reliability import (
    arrhenius_acceleration_factor,
    exponential_conditional_failure_probability,
    weibull_conditional_failure_probability,
)

logger = logging.getLogger(__name__)

_LINE_SEGMENT_LEN = 8  # cells per row/column fault segment, distribution="line"


def inject_faults(
    handle: CrossbarHandle,
    monitor: HealthMonitor,
    fault_config: FaultConfig,
    generator: torch.Generator | None = None,
) -> int:
    """generator: optional dedicated RNG used for EVERY random draw below
    (randperm / multinomial / rand / bernoulli). None (default) passes
    generator=None to each torch call, which is exactly the global-RNG
    consumption of the pre-generator code - byte-identical (verified in
    tests/invariants/test_remap_capacity_and_refailure.py). A non-None
    generator must live on the same device as the crossbar tensors (torch
    requirement); this is what lets a caller give different mitigation arms
    identical fault maps regardless of how much global RNG each arm consumed.

    Also records which positions this call hit as monitor.last_injected_mask
    (reset to all-False at the start of every call, every failure model) and
    monitor.last_injected_is_stuck (False for drift: a drifted cell is not
    stuck, so a hit remapped spare is not a failed spare). system.py uses
    these to model spare re-failure."""
    monitor.last_injected_mask = torch.zeros_like(monitor.stuck_mask)
    monitor.last_injected_is_stuck = fault_config.failure_model != "drift"
    if fault_config.failure_model == "density":
        return _custom_stuck_at(handle, monitor, fault_config, generator)
    if fault_config.failure_model == "drift":
        return _drift(handle, monitor, fault_config, generator)
    return _statistical_stuck_at(handle, monitor, fault_config, generator)


def _custom_stuck_at(
    handle: CrossbarHandle,
    monitor: HealthMonitor,
    fault_config: FaultConfig,
    generator: torch.Generator | None = None,
) -> int:
    matrix = handle.accessor.read()
    rows, cols = matrix.shape
    total_devices = rows * cols
    num_faults = int(total_devices * fault_config.density)
    if num_faults == 0:
        return 0

    flat_indices = _pick_indices(
        rows, cols, num_faults, fault_config.distribution, matrix.device, generator
    )
    fault_rows = flat_indices // cols
    fault_cols = flat_indices % cols

    # stuck_lrs_fraction = share stuck at LRS (g_max); the default 0.5 gives the old n // 2 split
    half = int(len(flat_indices) * (1.0 - fault_config.stuck_lrs_fraction))
    lrs_rows, lrs_cols = fault_rows[:half], fault_cols[:half]  # stuck at HRS (low conductance)
    hrs_rows, hrs_cols = fault_rows[half:], fault_cols[half:]  # stuck at LRS (high conductance)

    matrix[lrs_rows, lrs_cols] = monitor.g_min
    matrix[hrs_rows, hrs_cols] = monitor.g_max
    handle.accessor.write(matrix)

    monitor.stuck_mask[fault_rows, fault_cols] = True
    monitor.last_injected_mask[fault_rows, fault_cols] = True
    monitor.health_scores[fault_rows, fault_cols] = 0
    monitor.stability_index[fault_rows, fault_cols] = 0.1
    monitor.stress[fault_rows, fault_cols] = 0.9

    logger.info(
        "Injected %d faults into %s (%s distribution)",
        len(flat_indices),
        handle.name,
        fault_config.distribution,
    )
    return len(flat_indices)


def _drift(
    handle: CrossbarHandle,
    monitor: HealthMonitor,
    fault_config: FaultConfig,
    generator: torch.Generator | None = None,
) -> int:
    """A real, distinct failure mode found missing during development (see
    config.py's FaultConfig.drift_magnitude/drift_direction docstring):
    cells shift gradually toward one conductance rail instead of being
    pinned there outright - never marked stuck_mask, since the cell stays
    writable/functional, just imprecise. This is the failure mode
    mitigation/soft.py's docstring says it corrects, which this project
    never actually generated before now.
    """
    matrix = handle.accessor.read()
    rows, cols = matrix.shape
    total_devices = rows * cols
    num_drifting = int(total_devices * fault_config.density)
    if num_drifting == 0:
        return 0

    flat_indices = _pick_indices(
        rows, cols, num_drifting, fault_config.distribution, matrix.device, generator
    )
    drift_rows = flat_indices // cols
    drift_cols = flat_indices % cols

    shift_magnitude = fault_config.drift_magnitude * monitor.g_range
    if fault_config.drift_direction == "toward_hrs":
        shift = torch.full_like(drift_rows, -1.0, dtype=torch.float32) * shift_magnitude
    elif fault_config.drift_direction == "toward_lrs":
        shift = torch.full_like(drift_rows, 1.0, dtype=torch.float32) * shift_magnitude
    else:  # "random_per_cell" - decided once per cell, reused thereafter
        sign = monitor.drift_sign[drift_rows, drift_cols]
        undecided = sign == 0
        if undecided.any():
            new_signs = torch.where(
                torch.rand(int(undecided.sum()), device=matrix.device, generator=generator) < 0.5,
                -1.0,
                1.0,
            )
            sign[undecided] = new_signs
            monitor.drift_sign[drift_rows, drift_cols] = sign
        shift = sign * shift_magnitude

    matrix[drift_rows, drift_cols] = torch.clamp(
        matrix[drift_rows, drift_cols] + shift, monitor.g_min, monitor.g_max
    )
    handle.accessor.write(matrix)
    monitor.drift_mask[drift_rows, drift_cols] = True
    monitor.last_injected_mask[drift_rows, drift_cols] = True

    logger.info(
        "Drifted %d cell(s) in %s toward %s (%s distribution, magnitude=%.3f)",
        len(flat_indices),
        handle.name,
        fault_config.drift_direction,
        fault_config.distribution,
        fault_config.drift_magnitude,
    )
    return len(flat_indices)


def _statistical_stuck_at(
    handle: CrossbarHandle,
    monitor: HealthMonitor,
    fault_config: FaultConfig,
    generator: torch.Generator | None = None,
) -> int:
    """Weibull/exponential time-to-failure fault injection - see module and
    faults/reliability.py docstrings. Vectorized: one torch.bernoulli() call
    over the whole matrix, no per-cell Python loop.

    "Surviving" means not effectively stuck: a remapped spare is a working
    (fresh, cycles reset by HealthMonitor.mark_remapped) device, so it keeps
    aging and can fail again, whereas a dead cell with no spare stops aging.
    With no remaps effective_stuck == stuck_mask, so behavior is unchanged.
    """
    dead = monitor.effective_stuck
    surviving = ~dead
    monitor.cycles[surviving] += 1

    acceleration = 1.0
    if fault_config.arrhenius_activation_energy_ev is not None:
        acceleration = arrhenius_acceleration_factor(
            fault_config.temperature_kelvin,
            fault_config.reference_temperature_kelvin,
            fault_config.arrhenius_activation_energy_ev,
        )

    delta_cycles = torch.full_like(monitor.cycles, acceleration)
    if fault_config.failure_model == "weibull":
        cycles_before = monitor.cycles - delta_cycles
        fail_prob = weibull_conditional_failure_probability(
            cycles_before, delta_cycles, fault_config.weibull_shape, fault_config.weibull_scale
        )
    elif fault_config.failure_model == "exponential":
        fail_prob = exponential_conditional_failure_probability(
            delta_cycles, fault_config.exponential_rate
        )
    else:
        raise ValueError(f"Unknown fault_config.failure_model: {fault_config.failure_model!r}")

    fail_prob = fail_prob.masked_fill(dead, 0.0)
    newly_failed = torch.bernoulli(fail_prob, generator=generator).bool()

    rows, cols = torch.nonzero(newly_failed, as_tuple=True)
    n_new = len(rows)
    if n_new == 0:
        return 0

    matrix = handle.accessor.read()
    # Each newly-failed cell independently stuck at one of the two extremes -
    # unlike _custom_stuck_at's fixed split by position, real stuck-at
    # physics doesn't spatially pattern which rail a device gets stuck at.
    stuck_at_min = torch.rand(n_new, device=matrix.device, generator=generator) < 0.5
    matrix[rows[stuck_at_min], cols[stuck_at_min]] = monitor.g_min
    matrix[rows[~stuck_at_min], cols[~stuck_at_min]] = monitor.g_max
    handle.accessor.write(matrix)

    monitor.stuck_mask[rows, cols] = True
    monitor.last_injected_mask[rows, cols] = True
    monitor.health_scores[rows, cols] = 0
    monitor.stability_index[rows, cols] = 0.1
    monitor.stress[rows, cols] = 0.9

    logger.info(
        "Injected %d faults into %s (%s failure model, max cycles=%d)",
        n_new,
        handle.name,
        fault_config.failure_model,
        int(monitor.cycles.max().item()),
    )
    return n_new


def _select_fault_indices(
    rows: int,
    cols: int,
    num_faults: int,
    distribution: str,
    device: torch.device,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    total = rows * cols
    num_faults = min(num_faults, total)

    if distribution == "clustered":
        num_centers = max(1, num_faults // 5)
        centers = torch.randperm(total, device=device, generator=generator)[:num_centers]
        center_rows, center_cols = centers // cols, centers % cols
        offsets = torch.tensor([(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1)], device=device)

        candidate_rows = (center_rows.unsqueeze(1) + offsets[:, 0].unsqueeze(0)).flatten()
        candidate_cols = (center_cols.unsqueeze(1) + offsets[:, 1].unsqueeze(0)).flatten()
        valid = (
            (candidate_rows >= 0)
            & (candidate_rows < rows)
            & (candidate_cols >= 0)
            & (candidate_cols < cols)
        )
        candidate_flat = (candidate_rows[valid] * cols + candidate_cols[valid]).unique()
        # Random subset: unique() sorts, so a plain [:num_faults] would keep only the lowest flat
        # indices (top rows) and bias clustered faults spatially.
        keep = torch.randperm(
            candidate_flat.numel(), device=candidate_flat.device, generator=generator
        )
        return candidate_flat[keep[:num_faults]]

    if distribution == "gradient":
        all_rows = torch.arange(rows, device=device).repeat_interleave(cols)
        all_cols = torch.arange(cols, device=device).repeat(rows)
        distances = all_rows.float() / rows + all_cols.float() / cols
        weights = torch.clamp(1 - distances / 2, min=1e-6)
        return torch.multinomial(weights, num_faults, replacement=False, generator=generator)

    return torch.randperm(total, device=device, generator=generator)[
        :num_faults
    ]  # "random" (default)


def _select_line_indices(
    rows: int,
    cols: int,
    num_faults: int,
    device: torch.device,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """distribution="line" selector. Lives beside _select_fault_indices (which stays
    unchanged because other code imports it)."""
    total = rows * cols
    num_faults = min(num_faults, total)
    # Row/column-correlated faults (word-line / bit-line defects): short straight segments
    # of _LINE_SEGMENT_LEN cells, each along a row or a column (50/50). Centers are
    # oversampled by 1.5 because segments overlap and clip at the edges; unique() then a
    # random subset keeps exactly num_faults when enough cells are covered (same pattern
    # as "clustered"). Opt-in only: no existing config uses it.
    n_seg = max(1, -(-int(num_faults * 1.5) // _LINE_SEGMENT_LEN))
    starts = torch.randperm(total, device=device, generator=generator)[:n_seg]
    s_rows, s_cols = starts // cols, starts % cols
    along_row = torch.rand(starts.numel(), device=device, generator=generator) < 0.5
    steps = torch.arange(_LINE_SEGMENT_LEN, device=device)
    cand_rows = s_rows.unsqueeze(1) + torch.where(along_row.unsqueeze(1), 0, steps.unsqueeze(0))
    cand_cols = s_cols.unsqueeze(1) + torch.where(along_row.unsqueeze(1), steps.unsqueeze(0), 0)
    cand_rows, cand_cols = cand_rows.flatten(), cand_cols.flatten()
    valid = (cand_rows < rows) & (cand_cols < cols)
    candidate_flat = (cand_rows[valid] * cols + cand_cols[valid]).unique()
    keep = torch.randperm(candidate_flat.numel(), device=candidate_flat.device, generator=generator)
    return candidate_flat[keep[:num_faults]]


def _pick_indices(
    rows: int,
    cols: int,
    num_faults: int,
    distribution: str,
    device: torch.device,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Route "line" to its own selector; every other distribution goes to the unchanged
    _select_fault_indices (same arguments, same RNG use)."""
    if distribution == "line":
        return _select_line_indices(rows, cols, num_faults, device, generator)
    return _select_fault_indices(rows, cols, num_faults, distribution, device, generator)
