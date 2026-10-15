"""Orchestrator: wires devices -> crossbar patching -> health monitoring ->
fault injection -> mitigation dispatch into one fault-tolerant model.

Builds exactly one CrossbarHandle, one HealthMonitor, and one redundancy pool
per memristive crossbar found after patching - and every downstream module
(faults, mitigation) is given that same handle, never a second copy. This is
the structural fix for the original codebase's bugs #1, #3 and #6: there is no code path anywhere in this
package that constructs a second Crossbar object for a layer that already
has one, and Conv2d layers are patched by default alongside Linear.
"""

from __future__ import annotations

import logging

import torch

from neurofault.config import ExperimentConfig
from neurofault.crossbar.health_monitor import HealthMonitor
from neurofault.crossbar.self_healing import (
    RedundancyPool,
    RowColumnRedundancyPool,
    build_redundancy_pool,
)
from neurofault.devices.registry import build_device
from neurofault.faults.injection import inject_faults
from neurofault.mitigation.dispatch import (
    MitigationOutcome,
    apply_runtime_mitigation,
    apply_stuck_remap,
)
from neurofault.mitigation.reset import apply_layer_reset

logger = logging.getLogger(__name__)


def _backend_module(simulator: str):
    """Lazily import the crossbar backend module for the given simulator
    name, so system.py never depends on a backend that isn't in use."""
    if simulator == "crosssim":
        from neurofault.crossbar.backends import crosssim_backend

        return crosssim_backend
    if simulator == "aihwkit":
        from neurofault.crossbar.backends import aihwkit_backend

        return aihwkit_backend
    raise ValueError(f"Unknown simulator backend: {simulator!r}")


class FaultTolerantNeuromorphic:
    def __init__(
        self,
        base_model: torch.nn.Module,
        config: ExperimentConfig,
        calibration_images: torch.Tensor | None = None,
    ):
        self.config = config
        backend = _backend_module(config.simulator)
        self._backend = backend

        if config.crossbar.calibrate_dac and config.simulator != "crosssim":
            # See CrossbarConfig.calibrate_dac's docstring - the fix
            # (CrossSim's native input-percentile-scaling mechanism) is CrossSim-
            # specific; a non-CrossSim backend would silently no-op this flag rather
            # than actually applying it, the same silent-footgun calibrate_adc's own
            # guard below already exists to prevent. Checked before build_device()
            # below so this is the error a caller sees, not an unrelated device-name
            # mismatch from passing a CrossSim-flavored config to the aihwkit path.
            raise ValueError(
                f"config.crossbar.calibrate_dac=True but backend {config.simulator!r} "
                "has no matching implementation (CrossSim-only)"
            )

        self.device_cls, self.device_params = build_device(config.device, config.simulator)

        calibrated_ranges = None
        if config.crossbar.calibrate_adc:
            # CrossSim-only (ADCRangeLimits.MAX fix) - a config
            # with calibrate_adc=True but a non-CrossSim backend, or no real
            # calibration_images supplied by the caller, is a caller error the
            # backend/experiments/run.py contract should already prevent, not
            # something to silently no-op past here.
            if not hasattr(backend, "calibrate_adc_ranges"):
                raise ValueError(
                    f"config.crossbar.calibrate_adc=True but backend {config.simulator!r} "
                    "has no calibrate_adc_ranges() (CrossSim-only)"
                )
            if calibration_images is None:
                raise ValueError(
                    "config.crossbar.calibrate_adc=True requires calibration_images "
                    "(see experiments/run.py for how a real calibration batch is sourced)"
                )
            calibrated_ranges = backend.calibrate_adc_ranges(
                base_model, config.crossbar, self.device_params, calibration_images
            )

        self.model = backend.patch_layers(
            base_model, config.crossbar, self.device_cls, self.device_params, calibrated_ranges
        )

        self.handles = backend.build_handles(self.model, self.device_params, config.crossbar)

        # Not every backend has an equivalent weight->conductance mapping for
        # re-deriving conductances from weights - layer_reset degrades to
        # "unavailable" (never a crash) for backends that don't define this.
        mapping_routine_for = getattr(backend, "mapping_routine_for", None)
        self._mapping_routine = (
            mapping_routine_for(config.crossbar.scheme) if mapping_routine_for else None
        )
        self.monitors = {name: HealthMonitor(h) for name, h in self.handles.items()}
        self.pools = {
            name: build_redundancy_pool(h, config.crossbar) for name, h in self.handles.items()
        }

        # Original weights per patched layer, captured once, for layer_reset.
        # Membership must be checked against self.handles (built above by the
        # backend, so it already names exactly the patched layers) - a prior
        # version of this check looked for a `.crossbars` attribute that no
        # patched layer in either backend (AnalogLinear/AnalogConv2d for
        # CrossSim, TorchSimulatorTile for AIHWKit) ever actually defines,
        # which silently left _layer_handles empty and layer_reset dead on
        # both backends regardless of config - a real bug found and fixed
        # during development, confirmed via a direct runtime probe on both.
        patched_layers = {h.layer for h in self.handles.values()}
        self._original_weights: dict[str, torch.Tensor] = {}
        self._layer_handles: dict[str, list] = {}
        for name, module in self.model.named_modules():
            if module in patched_layers and getattr(module, "weight", None) is not None:
                self._original_weights[name] = module.weight.data.clone()
                self._layer_handles[name] = [
                    h for hname, h in self.handles.items() if h.layer is module
                ]

        self.total_operations = 0
        self.total_faults_injected = 0
        self.stats = {"soft_mitigations": 0, "remappings": 0, "layer_resets": 0}

        # Conductance values that must be reasserted after every synchronize()
        # (see synchronize()/_reapply_pins() below) because they don't durably
        # survive being re-derived from .weight - a real bug found and fixed
        # during development: this used to be a fault-aware-retraining-only
        # mechanism (_capture_fixed_defects()/reapply_stuck_faults()), but
        # stuck-at faults silently self-healed within one batch in every
        # OTHER mode too, verified directly (86/86 stuck cells reverted after
        # one real optimizer.step()+synchronize()). Two categories:
        # - _stuck_pin_values: permanent, updated incrementally by
        #   inject_faults() below, for every mode.
        # - _pending_one_shot_pins: soft-mitigation's corrective nudge only -
        #   models drift correction, not a permanent structural change, so it
        #   survives exactly the one synchronize() that would otherwise erase
        #   it before any forward pass ever benefits from it, then releases
        #   so ordinary training resumes shaping that cell from the corrected
        #   baseline. Remap needs no equivalent entry here: its corrected
        #   positions are already tracked permanently by the pool itself
        #   (RedundancyPool.remapping / RowColumnRedundancyPool's
        #   condemned_rows/condemned_cols), reapplied directly from there.
        self._stuck_pin_values: dict[str, torch.Tensor] = {}
        self._pending_one_shot_pins: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}

        # layer_reset cooldown bookkeeping - see MitigationConfig.
        # layer_reset_cooldown_checks and _attempt_layer_reset()'s docstring.
        self._health_check_count = 0
        self._last_reset_check: dict[str, int] = {}

        # --- stuck-weight freeze, reference_mode, remap ordering -------------
        # Verified probes (CrossSim): backward already reads the PROGRAMMED matrix
        # (core.get_core_weights(), stuck pins included), so grad_x sees stuck values
        # correctly - but .weight at a stuck position keeps receiving grad/momentum/
        # weight-decay updates every step (measured ~1e-3/step, up to 0.38 off the
        # pinned value), drifting to garbage that a later remap would copy into the
        # spare. So each cell's .weight is snapshotted when it becomes effectively
        # stuck (its value-at-fault) and restored after every synchronize(). This is
        # behavior-neutral for every existing continuous metric: no other cell's
        # forward/backward reads a stuck cell's .weight) - with ONE measured exception:
        # CrossSim's synchronize() re-derives the WHOLE array's limits from .weight, so a
        # stuck cell whose drifting .weight leaves the matrix's [min, max] rescales (and
        # ulp-perturbs, ~1e-8 with ideal devices) every other cell. Freezing removes that
        # spurious coupling, which is the point - but it moves grad_norm by ~1e-7 relative
        # in the refactor golden (loss/accuracy/faults/health byte-equal). So the freeze is
        # AUTO-ON only in the deployment regime and off for the legacy continuous path
        # (retraining.freeze_stuck_weights=None), keeping items 1-3's results reproducible.
        # CrossSim only - the accessor-matrix == weight.reshape(out, -1) identity was
        # verified for that backend alone.
        freeze = config.retraining.freeze_stuck_weights
        if freeze is None:
            # continuous + a retraining policy needs it too (living spares copy the
            # value-at-fault weight); policy "none" stays off, byte-identical to items 1-3.
            freeze = config.regime == "deployment" or config.retraining.policy != "none"
        self._freeze_enabled = config.simulator == "crosssim" and freeze
        self._stuck_weight_snapshot: dict[str, torch.Tensor] = {}
        self._stuck_weight_valid: dict[str, torch.Tensor] = {}
        self.reference_mode = config.retraining.reference_mode
        if self.reference_mode in ("living", "living_orig") and not self._freeze_enabled:
            raise ValueError(
                "retraining.reference_mode='living' needs the value-at-fault snapshots of the "
                "stuck-weight freeze (CrossSim backend, retraining.freeze_stuck_weights true or "
                "regime='deployment')"
            )
        if self.reference_mode in ("living", "living_orig") and not all(
            isinstance(p, RedundancyPool) for p in self.pools.values()
        ):
            raise ValueError(
                "retraining.reference_mode='living' needs per-cell spares "
                "(crossbar.redundancy_scheme='idealized_per_cell')"
            )
        # Diagonal-Fisher tracker for remap_order='saliency' (set by the deployment
        # loop from the checkpoint sidecar); None until attached.
        self.fisher = None
        # Closed loop (spectral X.1): optional fault-map estimator; see fault_map_source
        # and localize_now(). None/"oracle" = every decision reads the injector's truth.
        self.fault_estimator = None
        self._localize_calls = 0
        self.localize_log: list[dict] = []
        # Continuous regime + a retraining policy: mitigation is driven ONLY by the policy
        # (remap_now at each wave), exactly as in a deployment burst - the default
        # health-gated dispatch in forward() must not remap/reset behind its back.
        self._policy_driven = config.regime == "continuous" and config.retraining.policy != "none"
        # Per-layer count of remap-selection calls, only used to derive a
        # deterministic, arm-independent seed for remap_order='random'.
        self._remap_calls: dict[str, int] = {}

        logger.info(
            "Initialized FaultTolerantNeuromorphic with %d memristive crossbar(s)",
            len(self.handles),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.set_grad_enabled(self.model.training):
            output = self.model(x)

        with torch.no_grad():
            self.total_operations += 1
            if (
                not self._policy_driven
                and self.total_operations % self.config.mitigation.health_check_interval == 0
            ):
                self.check_health()

        return output

    def synchronize(self) -> None:
        """Call after every optimizer.step() during training. Some backends
        (CrossSim - see crosssim_backend.py's module docstring) don't
        auto-propagate gradient updates into the analog compute path without
        this; others (AIHWKit) already do and expose a no-op here. Missing
        this call means training silently never reaches the crossbar state
        for backends that need it - a real bug found and fixed during development,
        not a defensive no-op added speculatively.

        Always followed by apply_pins(): synchronize_fn() re-derives every
        cell's conductance from .weight, which knows nothing about stuck-at
        faults or mitigation corrections - a real bug found and fixed during
        development (verified directly: 86/86 stuck cells and a 122-cell remap
        both got silently erased by exactly this call). This runs for every
        mode now, not just fault-aware retraining - see _stuck_pin_values'
        docstring in __init__.
        """
        synchronize_fn = getattr(self._backend, "synchronize", None)
        if synchronize_fn is not None:
            synchronize_fn(self.model)
        self.apply_pins()

    def _reapply_pins(self) -> None:
        self.apply_pins()

    @property
    def fault_map_source(self) -> str:
        """'oracle' (default): controller decisions read the injector's stuck_mask.
        'estimated': they read HealthMonitor.believed_stuck, written by localize_now()."""
        return next(iter(self.monitors.values())).fault_map_source if self.monitors else "oracle"

    @fault_map_source.setter
    def fault_map_source(self, source: str) -> None:
        if source not in ("oracle", "estimated"):
            raise ValueError(f"fault_map_source must be 'oracle' or 'estimated', got {source!r}")
        for monitor in self.monitors.values():
            monitor.fault_map_source = source

    def localize_now(self) -> dict:
        """Estimate every layer's new stuck cells from counted probe reads and add them to
        the controller's belief (HealthMonitor.believed_stuck / believed_deviation).

        The estimator only gets a provider of the layer's current matrix, which it must
        read through its own counted, noisy reader, and the controller's reference: the
        digital weights plus the deviations already believed for cells not yet repaired
        (so known faults are not re-found). Truth (stuck_mask) is used below for LOGGING
        only (precision/recall), never for the belief. Returns per-wave stats."""
        if self.fault_estimator is None:
            raise ValueError("localize_now() needs system.fault_estimator")
        call = self._localize_calls
        self._localize_calls += 1
        tot = {"reads": 0, "tiles": 0, "fallback_tiles": 0, "shadow_reads": 0,
               "believed_new": 0, "tp": 0, "fp": 0, "fn": 0, "fn_visible": 0,
               "truth_new": 0, "truth_new_visible": 0, "found_new_visible": 0,
               "shadow_found_new_visible": 0, "shadow_fp": 0}  # fmt: skip
        for name, handle in self.handles.items():
            monitor = self.monitors[name]
            weight = self._weight_matrix(handle)
            if weight is None:
                raise ValueError(f"localize_now() needs a 1:1 weight/crossbar layout ({name!r})")
            known = monitor.believed_stuck & ~monitor.remapped_mask
            w_ref = weight + torch.where(known, monitor.believed_deviation.to(weight.dtype), 0)
            est = self.fault_estimator(name, handle.accessor.read, w_ref, call)
            new = est.mask.to(monitor.device) & ~monitor.believed_stuck
            monitor.believed_deviation[new] = est.deviation.to(monitor.device)[new.cpu()].to(
                monitor.believed_deviation.dtype
            )
            before = monitor.believed_stuck.clone()
            monitor.believed_stuck |= new
            # ---- logging only (truth) ----
            truth = monitor.effective_stuck
            truth_new = truth & ~before
            pinned = self._stuck_pin_values.get(name)
            dev_true = (pinned - weight) if pinned is not None else torch.zeros_like(weight)
            visible = truth & (dev_true.abs() >= 0.1 * float(weight.abs().max()))
            dec = monitor.decision_stuck
            tot["reads"] += est.reads
            tot["tiles"] += est.tiles
            tot["fallback_tiles"] += est.fallback_tiles
            tot["believed_new"] += int(new.sum())
            tot["tp"] += int((dec & truth).sum())
            tot["fp"] += int((dec & ~truth).sum())
            tot["fn"] += int((truth & ~dec).sum())
            tot["fn_visible"] += int((visible & ~dec).sum())
            tot["truth_new"] += int(truth_new.sum())
            tot["truth_new_visible"] += int((truth_new & visible).sum())
            tot["found_new_visible"] += int((new & truth_new & visible).sum())
            if est.shadow:
                smask = est.shadow["mask"].to(monitor.device)
                tot["shadow_reads"] += est.shadow["reads"]
                tot["shadow_found_new_visible"] += int((smask & truth_new & visible).sum())
                tot["shadow_fp"] += int((smask & ~truth).sum())
        tot["call"] = call
        self.localize_log.append(tot)
        return tot

    def unremapped_decision_stuck(self) -> dict[str, int]:
        """Per layer: cells the controller believes are stuck and unrepaired (equals
        unremapped_effective_stuck under the oracle)."""
        return {name: int(m.decision_stuck.sum()) for name, m in self.monitors.items()}

    def _weight_matrix(self, handle) -> torch.Tensor | None:
        """The layer's .weight as a (rows, cols) VIEW in accessor-matrix layout, or
        None when the freeze mechanics don't apply (non-CrossSim backend, or a layer
        whose weight doesn't map 1:1 onto the crossbar)."""
        if not self._freeze_enabled:
            return None
        weight = getattr(handle.layer, "weight", None)
        rows, cols = handle.accessor.shape
        if weight is None or weight.numel() != rows * cols:
            return None
        return weight.data.view(rows, -1)

    def apply_pins(self) -> None:
        """Re-assert everything synchronize() erases (stuck values, remap pins,
        one-shot soft pins) on the conductances, then restore frozen .weight at
        effectively-stuck cells - synchronize() minus the backend sync. Public so the
        deployment loop can call it after an injection wave (no optimizer step, hence
        no synchronize(), has happened to make the fault visible yet).

        reference_mode='living': remapped spares are NOT pinned here. Their position
        stays in pool.remapping (capacity bookkeeping) but the spare trains like any
        cell, starting from the value-at-fault written by _on_remapped()."""
        for name, handle in self.handles.items():
            monitor = self.monitors[name]
            pool = self.pools[name]
            changed = False
            matrix = handle.accessor.read()

            stuck_values = self._stuck_pin_values.get(name)
            # effective_stuck: a remapped cell is pinned by the pool's remap pin
            # below, never by its (dead) device's stuck value. A spare that later
            # fails leaves pool.remapping and re-enters effective_stuck together
            # (see _fail_hit_spares), so exactly one of the two pins applies.
            effective_stuck = monitor.effective_stuck
            if stuck_values is not None and effective_stuck.any():
                matrix[effective_stuck] = stuck_values[effective_stuck]
                changed = True

            if isinstance(pool, RowColumnRedundancyPool):
                for row in pool.condemned_rows:
                    matrix[row, :] = pool.reference_conductance[row, :]
                    changed = True
                for col in pool.condemned_cols:
                    matrix[:, col] = pool.reference_conductance[:, col]
                    changed = True
            elif pool.remapping and self.reference_mode == "frozen":
                # Vectorized - see RedundancyPool.select_remappable's docstring for the same
                # class of bug (a per-cell Python loop doing one GPU tensor write each) found
                # and fixed once already in mitigation/remap.py's dispatch-time write. This is
                # a SECOND, separate instance of it: _reapply_pins runs after every
                # synchronize() call - i.e. every batch, not just at health-check cadence - and
                # re-pins the FULL accumulated pool.remapping set each time, not just newly
                # remapped cells. Once the first remap burst populates hundreds of thousands of
                # entries, the per-cell loop form made every subsequent batch pay that cost
                # again - confirmed live on a real GPU campaign run (2026-09-30): ~40s/batch
                # instead of ~0.4s/batch immediately after the first remap burst, an arm that
                # would have taken ~6 days instead of ~3 hours to finish.
                rows_idx = torch.tensor([row for row, _col in pool.remapping], device=matrix.device)
                cols_idx = torch.tensor([col for _row, col in pool.remapping], device=matrix.device)
                matrix[rows_idx, cols_idx] = pool.reference_conductance[rows_idx, cols_idx]
                changed = True

            pending = self._pending_one_shot_pins.pop(name, None)
            if pending is not None:
                rows, cols, values = pending
                matrix[rows, cols] = values
                changed = True

            if changed:
                handle.accessor.write(matrix)

        self._restore_frozen_weights()

    def _restore_frozen_weights(self) -> None:
        for name, snapshot in self._stuck_weight_snapshot.items():
            handle = self.handles[name]
            effective_stuck = self.monitors[name].effective_stuck
            weight = self._weight_matrix(handle)
            if weight is not None and effective_stuck.any():
                weight[effective_stuck] = snapshot[effective_stuck]

    def inject_faults(self, generator: torch.Generator | None = None) -> int:
        """generator: optional dedicated RNG for every random draw of this call
        (see faults/injection.py). None (default) consumes the global torch RNG
        exactly as before. A caller that wants identical fault maps across runs
        whose global RNG state differs (e.g. different mitigation arms) passes
        an identically-seeded generator per event."""
        total = 0
        for name, handle in self.handles.items():
            monitor = self.monitors[name]
            total += inject_faults(handle, monitor, self.config.fault, generator=generator)
            self._fail_hit_spares(name)
            if self.config.secondary_fault is not None:
                total += inject_faults(
                    handle, monitor, self.config.secondary_fault, generator=generator
                )
                self._fail_hit_spares(name)
            self._update_stuck_pins(name, handle)
        self.total_faults_injected += total
        return total

    def _fail_hit_spares(self, name: str) -> None:
        """Spares age too: if the injection call that just ran hit a position
        currently served by a remapped spare (stuck-type models only), that
        spare has failed. Its position becomes effectively stuck again (stuck
        value written by injection, pinned by _update_stuck_pins), the remap
        pin is dropped, and the position is remap-eligible again IF capacity
        remains - capacity is NOT refunded (RedundancyPool.spares_used is
        monotonic).

        Verified before relying on it: injection samples all logical positions
        independent of stuck_mask for the "density" model, so a remapped cell
        is re-hit with the same probability as any other; for weibull/
        exponential it ages from a fresh cycle count (HealthMonitor.
        mark_remapped) and fails via the same conditional-probability path.
        RowColumnRedundancyPool: re-failure is NOT modeled - condemned
        rows/columns stay repaired (see its docstring); its cells are never
        in remapped_mask, so this is a no-op for it. Drift hits never count:
        a drifted cell is not stuck.
        """
        monitor = self.monitors[name]
        pool = self.pools[name]
        if not monitor.last_injected_is_stuck or not hasattr(pool, "release_failed_spares"):
            return
        failed = monitor.last_injected_mask & monitor.remapped_mask
        if not failed.any():
            return
        rows_idx, cols_idx = torch.nonzero(failed, as_tuple=True)
        monitor.remapped_mask[failed] = False
        # The failed spare is a new stuck cell: its value-at-fault is its CURRENT
        # .weight (not the original device's stale snapshot) - invalidate so
        # _update_stuck_pins takes a fresh one.
        if name in self._stuck_weight_valid:
            self._stuck_weight_valid[name][failed] = False
        # Health needs no touch-up here: the stuck-type injection paths already
        # set health 0 / stress 0.9 / stability 0.1 on every hit cell.
        pool.release_failed_spares(list(zip(rows_idx.tolist(), cols_idx.tolist())))
        logger.info(
            "%d remapped spare(s) in %s failed (capacity not refunded)", len(rows_idx), name
        )

    def _update_stuck_pins(self, name: str, handle) -> None:
        """Snapshot the just-written conductance value of every currently-
        stuck cell, so _reapply_pins() can re-clamp it after every future
        synchronize(). Called after every inject_faults() call in every
        mode (not just fault-aware retraining's one-time injection) -
        incremental and idempotent, since a stuck cell's value doesn't change
        between injection and this capture.

        Uses effective_stuck: a remapped cell's current conductance is its
        spare's reference value, and snapshotting that as its "stuck value"
        would be wrong if the spare later fails (the re-failed cell's own
        injection write is captured here at that point, since by then it is
        effectively stuck again)."""
        monitor = self.monitors[name]
        stuck = monitor.effective_stuck
        if not stuck.any():
            return
        current = handle.accessor.read()
        pinned = self._stuck_pin_values.setdefault(name, torch.zeros_like(current))
        pinned[stuck] = current[stuck]

        # Value-at-fault snapshot, ONLY for newly effective-stuck cells: an existing
        # snapshot is never overwritten (the weight has been frozen since, and a
        # re-hit of an already-stuck cell is not a new fault). A re-failed spare was
        # invalidated in _fail_hit_spares and so counts as new here.
        weight = self._weight_matrix(handle)
        if weight is not None:
            snapshot = self._stuck_weight_snapshot.setdefault(name, torch.zeros_like(weight))
            valid = self._stuck_weight_valid.setdefault(
                name, torch.zeros(weight.shape, dtype=torch.bool, device=weight.device)
            )
            new = stuck & ~valid
            snapshot[new] = weight[new]
            valid |= new

    def check_health(self) -> dict[str, MitigationOutcome]:
        outcomes = {}
        mitigation_config = self.config.mitigation
        self._health_check_count += 1

        for name, handle in self.handles.items():
            monitor = self.monitors[name]
            monitor.update_health_metrics()

            # Perf fix, verified safe:
            # every mitigation action function (apply_soft_mitigation, apply_remap,
            # apply_layer_reset's positions=[] path) already no-ops on an empty
            # position list, so nothing below this check can have any observable
            # effect when a layer has zero verified faults - regardless of what
            # get_critical_devices()'s heuristic (health score / failure
            # probability / extreme conductance) flags. That heuristic set can be
            # enormous even fault-free: probed directly on a widened fc1 layer
            # (768x3072 = 2.36M cells), 261,889-337,546 cells (11-14%) were
            # flagged "critical" with zero faults injected anywhere, and
            # everything below (the stuck_positions filter, get_health_summary,
            # apply_runtime_mitigation) used to run in full against that list
            # every health_check_interval batches regardless - confirmed via
            # cProfile as check_health's dominant cost, growing with model size.
            if not monitor.effective_stuck.any() and not monitor.drift_mask.any():
                continue
            # Verified this actually matters, not just theoretically: on the
            # widened SimpleCNN at fault density 0.0, clean per-batch timing
            # (no cProfile overhead) went from 601.6ms/batch to 181.7ms/batch
            # after this guard - a 3.3x speedup, all from avoiding this exact
            # dead-end path every health_check_interval batches.

            critical = monitor.get_critical_devices(
                detection_method=mitigation_config.detection_method,
                checksum_z_threshold=mitigation_config.checksum_z_threshold,
            )
            if not critical:
                continue

            summary = monitor.get_health_summary()

            layer_name = self._find_layer_name(handle)
            reset_fn = None
            if layer_name is not None and self._mapping_routine is not None:
                # Precision fix found during development: `critical` is get_critical_devices()'s
                # broad heuristic set (health score / failure probability /
                # extreme conductance), which can include cells that are
                # simply legitimately-trained near an extreme value, not
                # actually faulty. layer_reset should only ever touch cells
                # confirmed permanently, physically dead - filter to
                # stuck_mask here, same fix applied to remapping in
                # dispatch.py. Resolved lazily, at call time, against
                # effective_stuck (not stuck_mask): remap now runs before the
                # reset branches in the same dispatch call, and a cell just
                # bypassed by a spare must not also be reset to its original
                # weight.
                reset_fn = lambda ln=layer_name, m=monitor, c=critical: self._attempt_layer_reset(
                    ln, m.filter_positions_by_mask(c, m.effective_stuck)
                )
            elif layer_name is not None:
                logger.debug(
                    "Layer reset unavailable for simulator=%s (no mapping_routine_for); "
                    "skipping reset for %s",
                    self.config.simulator,
                    layer_name,
                )

            outcome = apply_runtime_mitigation(
                handle,
                self.pools[name],
                monitor,
                critical,
                summary["avg_health"],
                mitigation_config,
                reset_fn=reset_fn,
                remap_selector=self._make_remap_selector(
                    name, self.config.retraining.remap_order, None
                ),
            )
            self._on_remapped(name, handle, outcome.remapped_positions)
            outcomes[name] = outcome
            self.stats["soft_mitigations"] += outcome.soft_mitigated
            self.stats["remappings"] += outcome.remapped
            self.stats["layer_resets"] += int(outcome.layer_reset)

            self._pin_soft_mitigation(name, handle, outcome)

        return outcomes

    # --- remap ordering, living references, policy-driven remap ----------------------

    def _make_remap_selector(self, name: str, order: str, budget: int | None):
        """Ordering/truncation applied to a layer's effective-stuck candidates before
        the pool spends spares (dispatch.apply_stuck_remap's remap_selector seam).
        None = identity (legacy prefix order, no cap).

        prefix: row-major (legacy; a strawman for new arms - the lowest rows get every
          spare). random: a permutation from a dedicated generator seeded from
          (config.seed, layer, per-layer call count) - never the global RNG, so a
          burst's RNG use can't change which cells get spares and every arm sees the
          same order. magnitude: descending |w_stuck - w_at_fault|. saliency:
          descending fisher * (w_stuck - w_at_fault)^2 (needs self.fisher; raises
          rather than silently falling back to another order). budget: keep at most
          that many (per layer) - spares not spent now stay available (pacing)."""
        if budget is not None and budget < 0:
            raise ValueError(f"remap budget must be >= 0 or None, got {budget!r}")
        if order == "saliency" and self.fisher is None:
            raise ValueError(
                "remap_order='saliency' needs a FisherTracker: attach one as "
                "system.fisher (the deployment loop loads it from the pre-deploy "
                "checkpoint sidecar)"
            )
        if order == "prefix":
            return None if budget is None else (lambda positions: positions[:budget])
        if order not in ("random", "magnitude", "saliency"):
            raise ValueError(f"Unknown remap order: {order!r}")

        call = self._remap_calls.get(name, 0)
        self._remap_calls[name] = call + 1
        layer_ordinal = list(self.handles).index(name)

        def select(positions):
            n = len(positions)
            if n == 0:
                return positions
            if order == "random":
                g = torch.Generator()
                g.manual_seed(
                    ((self.config.seed & 0xFFFFFFFF) << 24) ^ (layer_ordinal << 16) ^ call
                )
                perm = torch.randperm(n, generator=g)
            else:
                score = self._stuck_deviation(name, positions)
                score = score.abs() if order == "magnitude" else score.pow(2)
                if order == "saliency":
                    # Fisher is stored in weight shape; the matrix layout is (out, -1).
                    fisher = self.fisher.fisher[name].reshape(
                        self.handles[name].accessor.shape[0], -1
                    )
                    rows = torch.tensor([r for r, _c in positions], device=fisher.device)
                    cols = torch.tensor([c for _r, c in positions], device=fisher.device)
                    score = fisher[rows, cols].to(score.device) * score
                perm = torch.sort(score, descending=True, stable=True).indices.cpu()
            ordered = [positions[i] for i in perm.tolist()]
            return ordered if budget is None else ordered[:budget]

        return select

    def _stuck_deviation(self, name: str, positions) -> torch.Tensor:
        """w_stuck - w_at_fault per position (what the stuck cell now reads minus what
        it should have read)."""
        if self.fault_map_source == "estimated":
            dev = self.monitors[name].believed_deviation
            rows = torch.tensor([r for r, _c in positions], device=dev.device)
            cols = torch.tensor([c for _r, c in positions], device=dev.device)
            return dev[rows, cols]
        snapshot = self._stuck_weight_snapshot.get(name)
        pinned = self._stuck_pin_values.get(name)
        if snapshot is None or pinned is None:
            raise ValueError(
                f"remap ordering by deviation needs value-at-fault snapshots for {name!r} "
                "(CrossSim backend, after inject_faults)"
            )
        rows = torch.tensor([r for r, _c in positions], device=snapshot.device)
        cols = torch.tensor([c for _r, c in positions], device=snapshot.device)
        return (pinned[rows, cols] - snapshot[rows, cols]).to(snapshot.dtype)

    def _on_remapped(self, name: str, handle, positions: list[tuple[int, int]]) -> None:
        """Post-remap hook, run for every remap (periodic dispatch AND remap_now).

        frozen: nothing - the spare stays pinned to pool.reference_conductance by
          apply_pins() every synchronize() (legacy behavior).
        living: the spare receives the cell's value-at-fault weight, written into BOTH
          the conductance matrix and .weight (so matrix == weight there and training
          continues from the weight the cell had when it died), and apply_pins() no
          longer pins it. Done here, after the pool wrote the reference value, instead
          of threading a flag through mitigation/remap.py: one system-level hook covers
          both remap entry points and keeps the pool pure bookkeeping.
        living_orig (ablation of living): the spare is not pinned and trains like living, but
          it starts from the ORIGINAL reference value the pool just wrote (not from the
          value-at-fault), so .weight is set to that value. Separates "spare keeps
          training" from "spare starts from the learned value"."""
        if self.reference_mode not in ("living", "living_orig") or not positions:
            return
        if self.reference_mode == "living_orig":
            weight = self._weight_matrix(handle)
            if weight is None:
                raise ValueError("reference_mode='living_orig' needs the CrossSim weight tensor")
            pool = self.pools[name]
            matrix = handle.accessor.read()
            rows = torch.tensor([r for r, _c in positions], device=matrix.device)
            cols = torch.tensor([c for _r, c in positions], device=matrix.device)
            values = pool.reference_conductance[rows, cols].to(matrix.dtype)
            matrix[rows, cols] = values
            handle.accessor.write(matrix)
            weight[rows, cols] = values.to(weight.dtype)
            monitor = self.monitors[name]
            written = handle.accessor.read()[rows, cols]
            for past in monitor.g_history:
                past[rows, cols] = written
            return
        weight = self._weight_matrix(handle)
        snapshot = self._stuck_weight_snapshot.get(name)
        estimated = self.fault_map_source == "estimated"
        # estimated: the value comes from .weight, so a layer with no injected fault yet
        # (no snapshot) can still remap a believed (false-positive) cell
        if weight is None or (snapshot is None and not estimated):
            raise ValueError("reference_mode='living' needs CrossSim value-at-fault snapshots")
        matrix = handle.accessor.read()
        rows = torch.tensor([r for r, _c in positions], device=matrix.device)
        cols = torch.tensor([c for _r, c in positions], device=matrix.device)
        if estimated:
            # The controller's value-at-fault is its own digital weight (frozen at true
            # stuck cells, so equal to the snapshot there; unchanged at a false positive).
            values = weight[rows, cols].clone()
        else:
            values = snapshot[rows, cols]
        matrix[rows, cols] = values.to(matrix.dtype)
        handle.accessor.write(matrix)
        weight[rows, cols] = values
        # Keep the monitor's history consistent with the spare's actual value (same
        # reason HealthMonitor.mark_remapped overwrites it with the reference value).
        monitor = self.monitors[name]
        written = handle.accessor.read()[rows, cols]
        for past in monitor.g_history:
            past[rows, cols] = written

    def remap_now(
        self, order: str | None = None, budget: int | dict[str, int] | None = None
    ) -> dict[str, int]:
        """Policy-driven remap: spend spares on every layer's effective-stuck cells
        immediately, in `order` (default config.retraining.remap_order), at most
        `budget` per layer (an int applies to every layer; a {layer: int} dict gives each
        layer its own cap - pacing needs this because spare pools are per layer and
        drain at different rates; a layer missing from the dict is uncapped). Deployment policies call this directly instead of relying
        on forward()'s every-N-calls check_health(), which never runs there (bursts and
        evals call system.model, not system.forward) and would couple remap timing to
        an unrelated cadence knob. Returns {layer: cells remapped}; also counted in
        stats['remappings'].

        Needs mitigation.enable_remapping (a policy decides WHEN; the config still
        decides whether this arm may remap at all - validator checks the counter
        against the flag). Row/column pools remap by line, so an order or budget is
        meaningless there and rejected."""
        if not self.config.mitigation.enable_remapping:
            raise ValueError("remap_now() requires mitigation.enable_remapping=True")
        order = self.config.retraining.remap_order if order is None else order
        remapped: dict[str, int] = {}
        for name, handle in self.handles.items():
            monitor, pool = self.monitors[name], self.pools[name]
            layer_budget = budget.get(name) if isinstance(budget, dict) else budget
            if isinstance(pool, RowColumnRedundancyPool) and (
                order != "prefix" or layer_budget is not None
            ):
                raise ValueError("remap order/budget are only defined for per-cell spares")
            candidates = [(r, c) for r, c in torch.nonzero(monitor.decision_stuck).tolist()]
            n, positions = apply_stuck_remap(
                handle,
                pool,
                monitor,
                candidates,
                self._make_remap_selector(name, order, layer_budget),
            )
            self._on_remapped(name, handle, positions)
            remapped[name] = n
            self.stats["remappings"] += n
        return remapped

    def unremapped_effective_stuck(self) -> dict[str, int]:
        """Per layer: stuck cells currently NOT bypassed by a working spare."""
        return {name: int(m.effective_stuck.sum()) for name, m in self.monitors.items()}

    def _attempt_layer_reset(self, layer_name: str, positions: list[tuple[int, int]]) -> bool:
        """Gated per mitigation.layer_reset_cooldown_checks (see
        MitigationConfig's docstring for the real problem this closes):
        returns False without resetting anything if this layer was reset
        too recently, measured in check_health() calls, not batches - so it
        composes with health_check_interval instead of a separate knob.

        positions (the calling handle's critical devices) targets the reset
        at only those cells - see apply_layer_reset's docstring for why a
        whole-layer reset is a real, measured design flaw, not just a
        theoretical one."""
        last = self._last_reset_check.get(layer_name)
        cooldown = self.config.mitigation.layer_reset_cooldown_checks
        if last is not None and self._health_check_count - last < cooldown:
            return False

        ok = apply_layer_reset(
            self._layer_handles[layer_name][0].layer,
            self._layer_handles[layer_name],
            self._original_weights[layer_name],
            self._mapping_routine,
            self.device_params.get("r_on", 100.0),
            self.device_params.get("r_off", 10000.0),
            scheme=self.config.crossbar.scheme,
            positions=positions,
        )
        if ok:
            self._last_reset_check[layer_name] = self._health_check_count
        return ok

    def _pin_soft_mitigation(self, name: str, handle, outcome: MitigationOutcome) -> None:
        """Pin any soft-mitigated positions through exactly the next
        synchronize() - see its docstring - then release so ordinary
        training resumes shaping those cells from the corrected baseline."""
        if not outcome.soft_mitigated_positions:
            return
        matrix = handle.accessor.read()
        rows = torch.tensor([r for r, _c in outcome.soft_mitigated_positions], device=matrix.device)
        cols = torch.tensor([c for _r, c in outcome.soft_mitigated_positions], device=matrix.device)
        values = matrix[rows, cols].clone()
        self._pending_one_shot_pins[name] = (rows, cols, values)

    def get_health_report(self) -> dict:
        summaries = {name: m.get_health_summary() for name, m in self.monitors.items()}
        avg_health = (
            sum(s["avg_health"] for s in summaries.values()) / len(summaries)
            if summaries
            else 100.0
        )
        return {
            "total_operations": self.total_operations,
            "total_faults_injected": self.total_faults_injected,
            **self.stats,
            "avg_health": avg_health,
            "layer_health": summaries,
        }

    def _find_layer_name(self, handle) -> str | None:
        for layer_name, handles in self._layer_handles.items():
            if handle in handles:
                return layer_name
        return None
