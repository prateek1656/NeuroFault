"""Redundancy remapping: restore faulty devices to their pre-fault reference
conductance, via a RedundancyPool's bookkeeping and the canonical
ConductanceAccessor for the actual I/O.

This is the direct fix for the original codebase's bug #1: the write goes
through `handle.accessor` - the same interface health monitoring and fault
injection use for this exact crossbar - never a separate, disconnected
crossbar object.

apply_remap() is the single public entry point regardless of which
redundancy scheme built `pool` - neurofault.mitigation.dispatch.py calls it
generically and never needs to know which scheme is active; the branch on
pool type lives here instead. See neurofault.crossbar.self_healing's module
docstring for what each scheme actually models.
"""

from __future__ import annotations

import logging

import torch

from neurofault.crossbar.array import CrossbarHandle
from neurofault.crossbar.self_healing import RedundancyPool, RowColumnRedundancyPool

logger = logging.getLogger(__name__)


def apply_remap(
    handle: CrossbarHandle,
    pool: RedundancyPool | RowColumnRedundancyPool,
    critical_devices: list[tuple[int, int]],
) -> int:
    return apply_remap_with_positions(handle, pool, critical_devices)[0]


def apply_remap_with_positions(
    handle: CrossbarHandle,
    pool: RedundancyPool | RowColumnRedundancyPool,
    critical_devices: list[tuple[int, int]],
) -> tuple[int, list[tuple[int, int]]]:
    """Same as apply_remap() (whose int return is unchanged) but also returns
    the exact cell positions that were remapped, so dispatch can tell the
    HealthMonitor (mark_remapped). For RowColumnRedundancyPool the position
    list is empty: that scheme's unit is whole rows/columns, which are tracked
    by the pool's condemned_rows/cols, not as per-cell remaps (see its class
    docstring)."""
    if isinstance(pool, RowColumnRedundancyPool):
        return _apply_row_col_remap(handle, pool, critical_devices), []
    return _apply_per_cell_remap(handle, pool, critical_devices)


def _apply_per_cell_remap(
    handle: CrossbarHandle,
    pool: RedundancyPool,
    critical_devices: list[tuple[int, int]],
) -> tuple[int, list[tuple[int, int]]]:
    # Vectorized (see RedundancyPool.select_remappable's docstring for the real perf bug
    # this replaced - a Python loop doing one GPU tensor write per cell, ~9.7s/batch once
    # critical_devices reaches hundreds of thousands of entries on ResNet-9's largest
    # crossbars). select_remappable does the bookkeeping and returns exactly the positions
    # to write - one batched read-modify-write replaces the per-cell loop.
    selected = pool.select_remappable(critical_devices)
    remapped = len(selected)

    if remapped > 0:
        matrix = handle.accessor.read()
        rows_idx = torch.tensor([r for r, _c in selected], device=matrix.device)
        cols_idx = torch.tensor([c for _r, c in selected], device=matrix.device)
        matrix[rows_idx, cols_idx] = pool.reference_conductance[rows_idx, cols_idx]
        handle.accessor.write(matrix)

    logger.info(
        "Remapped %d/%d devices in %s (%d capacity remaining)",
        remapped,
        len(critical_devices),
        handle.name,
        pool.available_capacity,
    )
    return remapped, selected


def _apply_row_col_remap(
    handle: CrossbarHandle,
    pool: RowColumnRedundancyPool,
    critical_devices: list[tuple[int, int]],
) -> int:
    """Restores whole condemned rows/columns to reference_conductance in one
    write. Returns len(rows)+len(cols) - the honest unit for this scheme is
    rows/columns repaired, not incidental cell count (a condemned row/column
    touches many cells at once, so a cell-count return would misrepresent
    what "one remapping" means here vs. the per-cell scheme).
    """
    new_rows, new_cols = pool.condemn_rows_and_cols(critical_devices)
    if not new_rows and not new_cols:
        return 0

    matrix = handle.accessor.read()
    for row in new_rows:
        matrix[row, :] = pool.reference_conductance[row, :]
    for col in new_cols:
        matrix[:, col] = pool.reference_conductance[:, col]
    handle.accessor.write(matrix)

    logger.info(
        "Condemned %d row(s), %d column(s) in %s (%d row / %d col capacity remaining)",
        len(new_rows),
        len(new_cols),
        handle.name,
        pool.available_row_capacity,
        pool.available_col_capacity,
    )
    return len(new_rows) + len(new_cols)
