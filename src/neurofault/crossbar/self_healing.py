"""Redundancy bookkeeping for one crossbar.

Pure bookkeeping only - no conductance I/O happens here. This never touches
a second, disconnected crossbar (the original codebase's bug #1); it only
tracks which cells/rows/columns have been "remapped" and how much capacity
remains. The actual read/restore-write against the canonical CrossbarHandle
happens in neurofault.mitigation.remap, via the ConductanceAccessor - so
these classes work identically regardless of which backend (crosssim/
aihwkit) that accessor wraps.

Two redundancy schemes (CrossbarConfig.redundancy_scheme), built via
build_redundancy_pool() below:

- RedundancyPool ("idealized_per_cell"): an idealized, software-only upper
  bound - "restore a faulty cell to its pre-fault reference value" - not
  physical spare row/column hardware. Documented scope decision, not an
  oversight.
- RowColumnRedundancyPool ("row_col_granularity"): repairs whole rows/columns
  at once - the granularity real designs actually repair at (word-line/
  bit-line, via the peripheral decoder) - using the *existing* logical
  crossbar, not physical spare-hardware capacity padded onto it. True
  physical spare-row/column hardware (padding a layer's crossbar dimensions
  beyond its logical weight size, with an index remap when one is condemned)
  would require both backends' patch_layers() to allocate larger-than-logical
  arrays - a real cross-backend change, still deferred. This class is the
  honest middle ground: right granularity, not yet right physical model.
"""

from __future__ import annotations

import logging
from collections import Counter

from neurofault.config import CrossbarConfig
from neurofault.crossbar.array import CrossbarHandle

logger = logging.getLogger(__name__)


class RedundancyPool:
    def __init__(self, handle: CrossbarHandle, redundancy_factor: float):
        self.handle = handle
        self.redundancy_factor = redundancy_factor
        rows, cols = handle.accessor.shape
        total_cells = rows * cols

        self.capacity = max(1, int(total_cells * redundancy_factor))
        # Currently-live remaps (reporting + the position set _reapply_pins() re-pins
        # every batch). NOT the capacity ledger: a spare that itself fails is dropped
        # from here (release_failed_spares) but stays consumed, so capacity is tracked
        # by spares_used below, never len(remapping).
        self.remapping: dict[tuple[int, int], tuple[int, int]] = {}
        # Spares ever consumed. Monotonic: a failed spare is not refunded - the
        # physical spare device is dead, and refunding would let a long wear-out
        # lifetime remap the same finite spare pool indefinitely.
        self.spares_used = 0
        # Reference conductance captured once, before any faults - the "known good"
        # value a remap restores. Read once via the accessor at construction time;
        # never re-derived from a second crossbar object.
        self.reference_conductance = handle.accessor.read().clone().detach()

        logger.info(
            "RedundancyPool for %s: capacity=%d (idealized_per_cell, %.0f%% of %d cells)",
            handle.name,
            self.capacity,
            redundancy_factor * 100,
            total_cells,
        )

    @property
    def available_capacity(self) -> int:
        return self.capacity - self.spares_used

    def remap_device(self, row: int, col: int) -> bool:
        """Record that (row, col) has been remapped, if capacity allows.

        Pure bookkeeping - does not touch the crossbar. Callers (see
        neurofault.mitigation.remap.apply_remap) are responsible for actually
        restoring the conductance value via the accessor once this returns True.
        """
        pos = (row, col)
        if pos in self.remapping:
            return False
        if self.available_capacity <= 0:
            return False

        self.remapping[pos] = pos  # kept for reporting; see module docstring
        self.spares_used += 1
        return True

    def release_failed_spares(self, positions: list[tuple[int, int]]) -> int:
        """Drop positions whose spare has itself failed from the live remap set,
        WITHOUT refunding capacity (spares_used is untouched). The position is
        then eligible for remapping again, consuming a fresh spare if any remain.
        Returns how many positions were actually live remaps."""
        released = 0
        for pos in positions:
            if self.remapping.pop(pos, None) is not None:
                released += 1
        return released

    def select_remappable(self, positions: list[tuple[int, int]]) -> list[tuple[int, int]]:
        """Batch form of remap_device: returns the prefix of `positions` that
        gets newly recorded as remapped, and records it - same stop-the-whole-
        batch-at-the-first-failure semantics as calling remap_device() in a loop
        and breaking on the first False (an already-remapped position, or
        capacity exhausted), but as one dict scan instead of one Python-level
        call per position. Real perf bug found and fixed 2026-09-29: neurofault.mitigation.remap used
        to call remap_device() in a loop AND write the crossbar one cell at a
        time - on ResNet-9's largest crossbars, critical_devices routinely
        reaches hundreds of thousands of entries (see dispatch.py), and each
        per-cell tensor write is a separate GPU kernel launch - ~9.7s/batch at
        14% GPU utilization, confirmed via local CPU benchmark (0.59s for
        ~354K positions in the old per-call form) extrapolating almost exactly
        to the observed slowdown once GPU kernel-launch overhead is added.

        Prefix semantics are deliberately unchanged. The 2026-10-02 "selection
        stall" bug (every wave after the first remapped 0 cells with capacity
        left) was fixed upstream, not here: dispatch.py now filters
        already-remapped positions out of `positions` (HealthMonitor.
        effective_stuck), so the stop-at-first-already-remapped rule only
        remains as a guard against double-spending a spare.
        """
        stop_at = len(positions)
        for i, pos in enumerate(positions):
            if pos in self.remapping:
                stop_at = i
                break
        can_remap = min(stop_at, self.available_capacity)
        selected = positions[:can_remap]
        for pos in selected:
            self.remapping[pos] = pos
        self.spares_used += len(selected)
        return selected


class RowColumnRedundancyPool:
    """Repairs whole rows/columns at once instead of individual cells - see
    module docstring for why this is the right granularity without yet being
    the right physical model (no padded spare-hardware capacity).

    Separate spare-row and spare-column capacity, matching how real designs
    (ISAAC/PRIME-style) carry independent row and column sparing rather than
    one shared pool. redundancy_factor is reinterpreted per-axis here - a
    deliberate semantic shift from "fraction of cells" (RedundancyPool) to
    "fraction of rows" / "fraction of columns" - not a silent reuse.

    Spare re-failure (see system.py's inject_faults): NOT modeled here.
    Condemned rows/columns stay condemned for the run, and cells inside them
    keep stuck_mask=True / are never marked in HealthMonitor.remapped_mask
    (the repair is a whole-line restore, not a per-cell spare). Re-modeling
    "a repaired line's cells can fail again" would need per-cell aging of the
    spare line; deliberately left as the simplest honest choice - this scheme
    is an upper-bound repair model, not a wear-out lifetime model.
    """

    def __init__(
        self,
        handle: CrossbarHandle,
        redundancy_factor: float,
        condemn_threshold: float,
    ):
        self.handle = handle
        self.condemn_threshold = condemn_threshold
        rows, cols = handle.accessor.shape
        self.rows = rows
        self.cols = cols

        self.row_capacity = max(1, int(rows * redundancy_factor))
        self.col_capacity = max(1, int(cols * redundancy_factor))
        self.condemned_rows: set[int] = set()
        self.condemned_cols: set[int] = set()
        self.reference_conductance = handle.accessor.read().clone().detach()

        logger.info(
            "RowColumnRedundancyPool for %s: row_capacity=%d, col_capacity=%d "
            "(row_col_granularity, %.0f%% of %d rows / %d cols, condemn_threshold=%.2f)",
            handle.name,
            self.row_capacity,
            self.col_capacity,
            redundancy_factor * 100,
            rows,
            cols,
            condemn_threshold,
        )

    @property
    def available_row_capacity(self) -> int:
        return self.row_capacity - len(self.condemned_rows)

    @property
    def available_col_capacity(self) -> int:
        return self.col_capacity - len(self.condemned_cols)

    def condemn_rows_and_cols(
        self, critical_devices: list[tuple[int, int]]
    ) -> tuple[list[int], list[int]]:
        """Decide which rows/columns newly cross condemn_threshold given this
        batch of critical devices, consume capacity for them, and return the
        newly-condemned (not previously condemned) row and column indices.

        Pure bookkeeping - does not touch the crossbar. Callers (see
        neurofault.mitigation.remap.apply_remap) restore the returned rows/
        columns to reference_conductance via the accessor. Already-condemned
        rows/columns are never reconsidered - permanent for the run, mirroring
        RedundancyPool's one-time-consumption semantics.
        """
        row_counts = Counter(row for row, _col in critical_devices)
        col_counts = Counter(col for _row, col in critical_devices)

        candidate_rows = sorted(
            (
                row
                for row, count in row_counts.items()
                if row not in self.condemned_rows and count / self.cols >= self.condemn_threshold
            ),
            key=lambda row: -row_counts[row],
        )
        candidate_cols = sorted(
            (
                col
                for col, count in col_counts.items()
                if col not in self.condemned_cols and count / self.rows >= self.condemn_threshold
            ),
            key=lambda col: -col_counts[col],
        )

        new_rows = candidate_rows[: self.available_row_capacity]
        new_cols = candidate_cols[: self.available_col_capacity]
        self.condemned_rows.update(new_rows)
        self.condemned_cols.update(new_cols)
        return new_rows, new_cols


def build_redundancy_pool(handle: CrossbarHandle, crossbar_config: CrossbarConfig):
    """Returns the redundancy pool for crossbar_config.redundancy_scheme -
    same dispatch-by-config-value pattern as devices/registry.py::build_device,
    so callers (system.py) never branch on scheme themselves."""
    if crossbar_config.redundancy_scheme == "row_col_granularity":
        return RowColumnRedundancyPool(
            handle, crossbar_config.redundancy_factor, crossbar_config.redundancy_condemn_threshold
        )
    if crossbar_config.redundancy_scheme == "idealized_per_cell":
        return RedundancyPool(handle, crossbar_config.redundancy_factor)
    # Real bug found and fixed during development: this used to be an unconditional
    # `else` - any unrecognized string (a typo included) silently fell back
    # to idealized_per_cell, no error. Verified directly:
    # redundancy_scheme="row_col_granularityy" (one extra letter) silently
    # produced a plain RedundancyPool. Fail loud instead, same standard as
    # devices/registry.py's device-name dispatch.
    raise ValueError(
        f"Unknown crossbar_config.redundancy_scheme: {crossbar_config.redundancy_scheme!r} "
        "(expected 'idealized_per_cell' or 'row_col_granularity')"
    )
