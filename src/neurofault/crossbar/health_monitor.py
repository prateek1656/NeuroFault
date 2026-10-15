"""Health monitoring for a single memristive crossbar.

Ported from the original health_monitor.py, with get_critical_devices()
rewritten as vectorized tensor ops instead of a Python double for-loop over
every (row, col) cell - that loop is confirmed (via notebook investigation)
to have actually caused a real training run to be killed for being too slow.

get_critical_devices() supports two detection methods (MitigationConfig.
detection_method): "full_diff" (default, always correct - reads every cell)
and "checksum" (a cheap row/column-sum first pass that falls back to the
same full_diff scan whenever a mismatch is found, so fault localization is
identical whenever a fault actually exists - only the no-fault case is
cheaper). Honesty note: this simulation doesn't model real crossbar
peripheral-readout circuit cost, so "cheaper" here means modeled cell-touches
(total_detection_cost/last_detection_cost, surfaced via get_health_summary()),
not actual wall-clock time - both methods cost one vectorized torch call
either way in this simulation. Checksums also have a real, known blind spot:
two faults in the same row/column that shift the sum in exactly-canceling
directions go undetected - tested explicitly in tests/unit/test_health_monitor.py,
not just mentioned.
"""

from __future__ import annotations

import logging

import torch

from neurofault.crossbar.array import CrossbarHandle

logger = logging.getLogger(__name__)


class HealthMonitor:
    """Tracks per-device stress/health/stability for one crossbar, and predicts
    which devices are likely heading toward failure."""

    def __init__(self, handle: CrossbarHandle):
        self.handle = handle
        self.shape = tuple(handle.accessor.shape)
        # Every tensor this class owns must live on the same device as the
        # crossbar's real conductance matrix, or any op mixing the two
        # raises (found via a real GPU smoke test - this class was never exercised on CUDA before
        # that). accessor.read() is the actual analog
        # layer's tensor, so its device is the correct source of truth.
        self.device = handle.accessor.read().device

        self.r_on = float(handle.device_params.get("r_on", 100.0))
        self.r_off = float(handle.device_params.get("r_off", 16000.0))
        self.g_min = 1.0 / self.r_off
        self.g_max = 1.0 / self.r_on
        self.g_range = self.g_max - self.g_min

        self.stress = torch.zeros(self.shape, device=self.device)
        self.health_scores = torch.ones(self.shape, device=self.device) * 100
        self.stability_index = torch.ones(self.shape, device=self.device)
        self.operation_counts = torch.zeros(self.shape, device=self.device)

        # Used by faults/injection.py's statistical failure models
        # (failure_model="weibull"/"exponential"), see that module and
        # faults/reliability.py. cycles is a simulation-time proxy - one
        # increment per inject_faults() call a cell survives, not a count of
        # literal physical read/write operations. stuck_mask marks cells
        # already permanently failed, so they stop aging and are never
        # resampled/re-failed - matches real stuck-at fault physics.
        self.cycles = torch.zeros(self.shape, device=self.device)
        self.stuck_mask = torch.zeros(self.shape, dtype=torch.bool, device=self.device)

        # Cells whose stuck defect has been physically bypassed by a spare
        # (see mark_remapped() and effective_stuck below). stuck_mask itself
        # stays True there on purpose - it is the record of "this logical
        # position's original device failed" and is what fault statistics and
        # the stuck pin snapshot key off - but every consumer that asks "is
        # this cell currently dead and in need of mitigation?" must use
        # effective_stuck instead. Real bug this closes (2026-10-02, verified
        # from a real run.log: wave 1 "Remapped 14745/14745 (7373 remaining)",
        # wave 2 "Remapped 0/14745"): remapped cells stayed in stuck_mask, so
        # dispatch's stuck_capable list began with already-remapped positions
        # and select_remappable()'s stop-at-first-already-remapped prefix rule
        # stopped at index 0 - every later wave remapped nothing.
        self.remapped_mask = torch.zeros(self.shape, dtype=torch.bool, device=self.device)

        # Closed loop (spectral X.1): what the controller BELIEVES is stuck,
        # written by system.localize_now() from counted probe reads. Only consulted when
        # fault_map_source == "estimated" (decision_stuck below); the default "oracle"
        # leaves every decision on the injector's stuck_mask, byte-for-byte as before.
        # Physics (pins, frozen weights, aging) always stays on the truth.
        self.fault_map_source = "oracle"
        self.believed_stuck = torch.zeros(self.shape, dtype=torch.bool, device=self.device)
        self.believed_deviation = torch.zeros(self.shape, device=self.device)

        # Positions hit by the most recent faults/injection.py call (every
        # failure model, drift included - see last_injected_is_stuck). A
        # position being hit does not on its own imply anything new for an
        # already-stuck cell; system.inject_faults() uses this to detect a
        # *remapped spare* being hit (spares age and fail too).
        self.last_injected_mask = torch.zeros(self.shape, dtype=torch.bool, device=self.device)
        self.last_injected_is_stuck = False

        # Used by faults/injection.py's failure_model="drift" (see
        # config.py's FaultConfig.drift_magnitude/drift_direction docstring -
        # a real, distinct failure mode found missing during development:
        # soft_mitigation targets drift specifically, but this project never
        # generated any until now). drift_mask marks cells that have ever
        # drifted - unlike stuck_mask, these cells stay writable/functional,
        # just imprecise, and dispatch.py targets soft_mitigation at exactly
        # this mask rather than the broader (and, per the development-time
        # findings, over-inclusive) get_critical_devices() set. drift_sign
        # is only consulted for drift_direction="random_per_cell": each
        # affected cell's drift direction is decided once and reused for
        # every subsequent event (0 = undecided, +1/-1 once assigned) -
        # real retention drift is monotonic per-device, not a coin flip
        # every injection call.
        self.drift_mask = torch.zeros(self.shape, dtype=torch.bool, device=self.device)
        self.drift_sign = torch.zeros(self.shape, device=self.device)

        self.g_history: list[torch.Tensor] = [handle.accessor.read().clone()]
        self.max_history = 20

        # Reference checksums for the "checksum" detection method - reuse
        # the pristine matrix already captured above, no new snapshot.
        self.reference_row_checksum = self.g_history[0].sum(dim=1)
        self.reference_col_checksum = self.g_history[0].sum(dim=0)
        self.total_detection_cost = 0
        self.last_detection_cost = 0

        self.failure_probability = torch.zeros(self.shape, device=self.device)

        logger.info("Initialized health monitor for %dx%d crossbar", *self.shape)

    @property
    def effective_stuck(self) -> torch.Tensor:
        """Cells that are stuck AND not currently bypassed by a working spare -
        the set mitigation (remap, layer_reset) should still act on, and the
        set whose health/stress must stay frozen at their injected values."""
        return self.stuck_mask & ~self.remapped_mask

    @property
    def decision_stuck(self) -> torch.Tensor:
        """Cells the CONTROLLER should treat as dead and unrepaired: effective_stuck under
        the oracle (default), the estimated belief minus repaired cells otherwise."""
        if self.fault_map_source == "estimated":
            return self.believed_stuck & ~self.remapped_mask
        return self.effective_stuck

    def mark_remapped(self, rows_idx, cols_idx) -> None:
        """Record that the stuck cells at (rows_idx, cols_idx) are now served
        by a working spare, and give that spare a clean bill of health.

        Resets health_scores to 100, stress to 0, stability_index to 1 (a
        spare is a fresh device - leaving the failed device's health=0/
        stress=0.9 would make get_critical_devices() flag a perfectly working
        cell as critical forever), and cycles to 0 so weibull/exponential
        aging (injection.py) starts the spare from a fresh age rather than
        the failed device's accumulated cycles. g_history at these positions
        is overwritten with the spare's current conductance: otherwise the
        first update_health_metrics() after the remap sees a rail-to-reference
        jump as a huge "change acceleration" and collapses stability_index to 0
        for one check (verified arithmetic: change_acceleration ~1 gives
        stability 0.9*1 + (1-10)*0.1 = 0) - an artifact of the bookkeeping,
        not instability of the spare.
        """
        rows_idx = torch.as_tensor(rows_idx, dtype=torch.long, device=self.device)
        cols_idx = torch.as_tensor(cols_idx, dtype=torch.long, device=self.device)
        if rows_idx.numel() == 0:
            return
        self.remapped_mask[rows_idx, cols_idx] = True
        self.health_scores[rows_idx, cols_idx] = 100
        self.stress[rows_idx, cols_idx] = 0
        self.stability_index[rows_idx, cols_idx] = 1
        self.cycles[rows_idx, cols_idx] = 0
        current = self.handle.accessor.read()
        values = current[rows_idx, cols_idx]
        for past in self.g_history:
            past[rows_idx, cols_idx] = values

    def update_health_metrics(
        self, voltage_applied: torch.Tensor | None = None, write_op: bool = False
    ):
        """Real bug found and fixed during development: this function used to
        recompute stress/health_scores/stability_index for every cell,
        stuck_mask included - a permanently-stuck cell's own conductance
        never changes, so its deviation/instability terms settle into a
        small steady-state stress value, and `stress = stress*0.99 +
        new_stress` decays *toward* that steady state every call regardless
        of the cell being dead forever. Verified directly: a cell injected
        with health_scores=0, stress=0.9 reached health_scores=74.73 after
        200 calls to this function - a permanently-failed cell silently
        reporting itself as mostly healthy. Fix: snapshot stuck-position
        values before the (unchanged) computation below, restore them after -
        stuck cells are now provably inert to this function, not just
        usually-inert in practice."""
        # effective_stuck, not stuck_mask: a remapped cell is served by a
        # working spare and must evolve like any healthy cell (see
        # mark_remapped()); freezing it at its reset values would be the
        # mirror-image bug of letting a dead cell decay toward "healthy".
        frozen = self.effective_stuck
        stuck_stress = self.stress[frozen].clone()
        stuck_health = self.health_scores[frozen].clone()
        stuck_stability = self.stability_index[frozen].clone()

        current_g = self.handle.accessor.read()
        self.operation_counts += 1

        g_normalized = torch.clamp((current_g - self.g_min) / self.g_range, 0, 1)
        deviation_from_mid = 2 * torch.abs(g_normalized - 0.5)

        if len(self.g_history) >= 3:
            prev_g1, prev_g2 = self.g_history[-1], self.g_history[-2]
            recent_change1 = torch.abs(current_g - prev_g1) / self.g_range
            recent_change2 = torch.abs(prev_g1 - prev_g2) / self.g_range
            change_acceleration = torch.abs(recent_change1 - recent_change2)

            self.stability_index = (
                self.stability_index * 0.9 + (1.0 - change_acceleration * 10) * 0.1
            )
            self.stability_index = torch.clamp(self.stability_index, 0.0, 1.0)

            unstable = recent_change1 > 0.1
            self.health_scores[unstable] -= 0.5

        op_stress_factor = 0.02 if write_op else 0.001
        voltage_factor = 1.0
        if voltage_applied is not None:
            voltage_factor = 1.0 + torch.abs(voltage_applied).mean().item()

        new_stress = op_stress_factor * voltage_factor * (0.5 + deviation_from_mid)
        new_stress = new_stress * (2.0 - self.stability_index)

        self.stress = torch.clamp(self.stress * 0.99 + new_stress, 0.0, 1.0)
        self.health_scores = torch.clamp(100 * (1.0 - self.stress), 0, 100)

        # Restore stuck cells - see this function's docstring for the bug
        # this closes. A stuck cell's conductance never changes, so nothing
        # above is meaningful for it; freezing it here, not skipping it
        # above, keeps the non-stuck-cell computation identically vectorized.
        self.stress[frozen] = stuck_stress
        self.health_scores[frozen] = stuck_health
        self.stability_index[frozen] = stuck_stability

        self.g_history.append(current_g.clone())
        if len(self.g_history) > self.max_history:
            self.g_history.pop(0)

    def predict_faults(self) -> torch.Tensor:
        g_normalized = torch.clamp((self.handle.accessor.read() - self.g_min) / self.g_range, 0, 1)
        proximity_to_extreme = torch.clamp(
            torch.max(4 * g_normalized**2, 4 * (1 - g_normalized) ** 2), 0.0, 1.0
        )
        instability_factor = 1.0 - self.stability_index

        drift_factor = torch.zeros(self.shape, device=self.device)
        if len(self.g_history) >= 5:
            g_diffs = torch.stack(
                [(self.g_history[-i] - self.g_history[-i - 1]) / self.g_range for i in range(1, 5)]
            )
            drift_magnitude = torch.mean(torch.abs(g_diffs), dim=0)
            drift_consistency = torch.std(torch.sign(g_diffs), dim=0)
            drift_factor = torch.clamp(drift_magnitude * (2.0 - drift_consistency) * 5.0, 0.0, 1.0)

        self.failure_probability = (
            0.4 * self.stress
            + 0.3 * instability_factor
            + 0.2 * proximity_to_extreme
            + 0.1 * drift_factor
        )
        return self.failure_probability

    def get_critical_devices(
        self,
        health_threshold: float = 10,
        probability_threshold: float = 0.8,
        extreme_threshold: float = 0.02,
        detection_method: str = "full_diff",
        checksum_z_threshold: float = 3.0,
    ) -> list[tuple[int, int]]:
        """Vectorized replacement for the original's Python double for-loop.

        Computes the full-array condition as one boolean mask, then converts
        to a Python list only over the (small) filtered result via
        torch.nonzero - never iterates every cell in Python.

        detection_method="checksum": cheap row/column-sum first pass (see
        module docstring). Returns [] immediately if no row/column checksum
        is flagged (modeled cost rows+cols) - otherwise falls through to the
        exact same full-diff scan below (modeled cost rows*cols), so a real
        fault is localized exactly as precisely as the default path.
        """
        if detection_method not in ("full_diff", "checksum"):
            # Real bug found and fixed during development: this dispatch used to
            # silently treat any unrecognized string as "full_diff" (the
            # `if detection_method == "checksum": ... ` block below simply
            # never triggered) - a typo'd config value ran indistinguishably
            # from a correctly-configured one. Fail loud instead.
            raise ValueError(
                f"Unknown detection_method: {detection_method!r} "
                "(expected 'full_diff' or 'checksum')"
            )

        rows, cols = self.shape
        if detection_method == "checksum":
            mismatched_rows, mismatched_cols = self._checksum_mismatch_rows_cols(
                checksum_z_threshold
            )
            if not mismatched_rows.any() and not mismatched_cols.any():
                self.last_detection_cost = rows + cols
                self.total_detection_cost += self.last_detection_cost
                return []

        self.last_detection_cost = rows * cols
        self.total_detection_cost += self.last_detection_cost

        self.predict_faults()
        g_normalized = torch.clamp((self.handle.accessor.read() - self.g_min) / self.g_range, 0, 1)
        mask = (
            (self.health_scores < health_threshold)
            | (self.failure_probability > probability_threshold)
            | (g_normalized < extreme_threshold)
            | (g_normalized > (1 - extreme_threshold))
        )
        rows_idx, cols_idx = torch.nonzero(mask, as_tuple=True)
        return list(zip(rows_idx.tolist(), cols_idx.tolist()))

    def filter_positions_by_mask(
        self, positions: list[tuple[int, int]], mask: torch.Tensor
    ) -> list[tuple[int, int]]:
        """Vectorized replacement for `[(r, c) for r, c in positions if mask[r, c]]`.

        Real perf bug found and fixed (2026-09-28, while planning the ResNet-9 fault-injection campaign):
        `system.py`'s layer_reset dispatch and `mitigation/dispatch.py`'s remap/
        soft_mitigation dispatch both used exactly that Python-loop-with-per-
        element-tensor-indexing pattern to filter get_critical_devices()'s
        result down to stuck_mask/drift_mask-verified cells. Never a
        correctness bug (get_critical_devices() itself is small at the small
        pre-ResNet9 crossbar sizes this was always tested at before), but a
        real, previously-unmeasured performance cliff once `positions` reaches
        hundreds of thousands of entries on ResNet-9's largest crossbars
        (2.36M cells): measured directly, ~1.47s of a ~1.66s check_health()
        call (~90%) at this project's real ~27%-stuck-fraction fault-campaign
        target, dwarfing get_critical_devices()'s own already-vectorized
        0.14s. One vectorized index (`mask[rows_idx, cols_idx]`) instead of
        len(positions) individual Python-level tensor scalar reads.
        """
        if not positions:
            return []
        rows_idx = torch.tensor([r for r, _c in positions], device=mask.device)
        cols_idx = torch.tensor([c for _r, c in positions], device=mask.device)
        keep = mask[rows_idx, cols_idx].tolist()
        return [pos for pos, keep_this in zip(positions, keep) if keep_this]

    def _checksum_mismatch_rows_cols(self, z_threshold: float) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (mismatched_rows, mismatched_cols) boolean masks.

        Real detection-method upgrade made during development: a row/column used to be flagged against a fixed
        fraction of g_range, the same absolute threshold regardless of how
        much checksums naturally fluctuate from ordinary training. Now
        adaptive (V-ABFT's approach, 2026): flag a row/column as an outlier
        relative to the whole population of row/column deviations this call,
        rather than a fixed magnitude always meaning "fault."

        Uses median/MAD (median absolute deviation), not mean/std - verified
        directly why this matters: a plain mean+z*std z-score is itself
        skewed by the very outlier it's trying to detect, especially in a
        small population (a single faulty row among 6 inflates the
        population's own std enough that the outlier no longer clears its
        own threshold). Median/MAD stays anchored to the majority as long as
        faults are a minority - the standard "modified z-score" convention
        (0.6745 scales MAD to be comparable to std for a normal
        distribution). When MAD is exactly 0 (the reference population has
        no spread at all - e.g. a freshly-constructed monitor before any
        training has touched any row), any nonzero deviation is anomalous by
        definition, handled explicitly rather than dividing by zero.

        Same honest limitation either way: self-referential to the current
        population, so it degrades once faults are a majority rather than a
        minority of rows/columns.
        """
        current = self.handle.accessor.read()
        row_deviation = (current.sum(dim=1) - self.reference_row_checksum).abs()
        col_deviation = (current.sum(dim=0) - self.reference_col_checksum).abs()

        row_mismatch = self._is_outlier(row_deviation, z_threshold)
        col_mismatch = self._is_outlier(col_deviation, z_threshold)
        return row_mismatch, col_mismatch

    @staticmethod
    def _is_outlier(deviation: torch.Tensor, z_threshold: float) -> torch.Tensor:
        median = deviation.median()
        mad = (deviation - median).abs().median()
        if mad == 0:
            return deviation > 0
        modified_z = 0.6745 * (deviation - median).abs() / mad
        return modified_z > z_threshold

    def get_health_summary(self) -> dict:
        self.predict_faults()
        # health_scores itself is always exactly in [0, 100] (clamped in
        # update_health_metrics()), but .mean() is a reduction, and GPU
        # reductions sum in a different (parallelized) order than CPU's
        # sequential sum - found via a real GPU run producing
        # 100.00000069358133 instead of exactly 100.0, a float-precision
        # artifact, not a real health value above the theoretical max.
        avg_health = min(100.0, max(0.0, self.health_scores.mean().item()))
        return {
            "avg_health": avg_health,
            "min_health": self.health_scores.min().item(),
            "avg_stability": self.stability_index.mean().item(),
            "avg_failure_prob": self.failure_probability.mean().item(),
            "critical_count": (self.health_scores < 20).sum().item(),
            "detection_cost": self.total_detection_cost,
        }
