"""Deployment-regime retraining policies (arms R0/R1/A/B/C/C' and the co-mitigation controller D).

Every policy answers two independent questions per wave, in this order:

1. STRUCTURAL: should spares be spent? That is decided by the config, NOT the policy:
   `mitigation.enable_remapping` true => `system.remap_now(order=remap_order)` runs
   right after injection for EVERY policy. Remap is structural hardware repair, so an
   arm that enables it gets it identically regardless of its retraining cadence (R1 =
   policy "none" + enable_remapping; C vs C' differ only in reference_mode). There is
   deliberately no "remap" policy enum value: it would duplicate the config flag and
   let the two disagree. The wave's remapped counts are recorded either way.
2. RETRAINING: should a burst run, and how long? That is the policy's only job:
   none: never. static (A): one fixed burst after wave 0 only. periodic (B): a fixed
   burst after every wave. capacity (C/C'): a fixed burst only when, after remapping,
   some layer still holds effectively-stuck cells (the wave exceeded spare capacity);
   when every stuck cell was bypassed the network is functionally intact (frozen
   references) or retrains itself in place (living), so no burst is spent.
   co_mitigation (D): owns BOTH steps (see CoMitigationPolicy), because its remap is
   paced and ordered per wave and its burst is gated on a post-BN-recalibration canary.

The per-wave "health" metric is a synthetic stress quantity that drifts with check
count even with zero faults, so it is never a trigger here.

Event fields every policy returns (merged into events.json by run_deployment):
burst_batches, optimizer_steps, stop_reason ("no_burst" | "fixed" | "plateau" |
"t_max"), canary_trace ([[batch, acc], ...]), remapped_this_wave,
remapped_by_layer, unremapped_effective_stuck.
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path

from neurofault.config import ExperimentConfig
from neurofault.retraining.bn_recal import recalibrate_bn
from neurofault.retraining.burst import evaluate_accuracy, run_burst
from neurofault.retraining.canary import canary_sigma
from neurofault.retraining.policy_hook import WaveContext
from neurofault.retraining.recovery_model import PatienceStopper

logger = logging.getLogger(__name__)


def _localize(system) -> dict | None:
    """Closed loop: refresh the controller's estimated fault map before it decides (no-op
    under the default oracle map)."""
    return system.localize_now() if system.fault_map_source == "estimated" else None


def _loc_fields(loc: dict | None, believed: int) -> dict:
    if loc is None:
        return {}
    return {"localize": loc, "unremapped_believed_stuck": believed}


class _BasePolicy:
    """Template: structural remap step, then `_wants_burst` -> fixed burst."""

    def _wants_burst(self, ctx: WaveContext, unremapped: int) -> bool:
        raise NotImplementedError

    def on_wave(self, ctx: WaveContext) -> dict:
        cfg = ctx.config
        r = cfg.retraining
        remapped_by_layer: dict[str, int] = {}
        loc = _localize(ctx.system)
        if cfg.mitigation.enable_remapping:
            remapped_by_layer = ctx.system.remap_now(order=r.remap_order)
        unremapped = sum(ctx.system.unremapped_effective_stuck().values())
        believed = sum(ctx.system.unremapped_decision_stuck().values())

        fields = {
            "burst_batches": 0,
            "optimizer_steps": 0,
            "stop_reason": "no_burst",
            "canary_trace": [],
            "remapped_this_wave": sum(remapped_by_layer.values()),
            "remapped_by_layer": remapped_by_layer,
            "unremapped_effective_stuck": unremapped,
            **_loc_fields(loc, believed),
        }
        if self._wants_burst(ctx, believed):
            result = run_burst(
                ctx.system,
                ctx.canary.burst_loader,
                lr=r.burst_lr,
                n_batches=r.burst_batches,
                canary=ctx.canary.canary_loader,
                eval_every=r.patience_eval_every,
                fisher=ctx.fisher,
                config=cfg,
                device=ctx.device,
            )
            fields.update(
                burst_batches=result.batches,
                optimizer_steps=result.optimizer_steps,
                stop_reason=result.stop_reason,
                canary_trace=result.canary_trace,
            )
        return fields


class NonePolicy(_BasePolicy):
    """R0 (no remap) / R1 (enable_remapping): never retrains."""

    def _wants_burst(self, ctx, unremapped):
        return False


class StaticPolicy(_BasePolicy):
    """A: one fixed burst after the first wave only."""

    def _wants_burst(self, ctx, unremapped):
        return ctx.wave_index == 0


class PeriodicPolicy(_BasePolicy):
    """B (and the compute-matched control when burst_batches is set to D's mean)."""

    def _wants_burst(self, ctx, unremapped):
        return True


class CapacityPolicy(_BasePolicy):
    """C / C': burst only when spare capacity could not absorb the wave."""

    def _wants_burst(self, ctx, unremapped):
        return unremapped > 0


# priors-file key -> RetrainingConfig field it overrides (scripts/calibrate_retrain_controller.py
# writes the keys on the left; "eta" is the burst learning rate).
_PRIOR_FIELDS = {
    "eta": "burst_lr",
    "delta": "canary_gate_delta",
    "kappa": "patience_min_gain",
    "k": "patience_window",
    "m": "patience_eval_every",
    "t_max": "burst_t_max",
}
# Pacing is deliberately NOT overlaid from priors even though the calibration script
# reports the simulator's preferred pacing: it is a structural component of D (and the
# thing the D-nopace ablation toggles), not a numeric threshold. Letting priors set it
# would silently turn D into its own ablation whenever the offline simulator - which
# models remap value as uniform per cell, i.e. cannot see saliency - prefers greedy.


def load_priors(path: str | Path) -> dict:
    """Read a calibrated-controller priors JSON. Raises (FileNotFoundError / ValueError)
    instead of returning defaults: D must never silently run uncalibrated."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(
            f"retraining.priors_path {str(p)!r} does not exist. It is produced by "
            "scripts/calibrate_retrain_controller.py --from-calibration-sweep after the "
            "GPU calibration sweep (experiments/calibration_sweep.py); policy "
            "'co_mitigation' must not run before it exists."
        )
    priors = json.loads(p.read_text())
    missing = [key for key in _PRIOR_FIELDS if key not in priors]
    if missing:
        raise ValueError(f"priors file {str(p)!r} lacks required keys {missing}")
    return priors


def effective_params(config: ExperimentConfig) -> tuple[dict, list[str]]:
    """The controller parameters actually used, as RetrainingConfig field names, plus
    the names of those taken from the priors file (empty when priors_path is None, in
    which case the config fields are authoritative)."""
    r = config.retraining
    params = {field: getattr(r, field) for field in _PRIOR_FIELDS.values()}
    params["remap_pacing"] = r.remap_pacing  # always from config, see _PRIOR_FIELDS
    from_priors: list[str] = []
    if r.priors_path is not None:
        priors = load_priors(r.priors_path)
        for key, field in _PRIOR_FIELDS.items():
            params[field] = priors[key]
            from_priors.append(field)
        # the same invariants RetrainingConfig enforces on hand-written values
        for field in ("patience_window", "patience_eval_every", "burst_t_max"):
            params[field] = int(params[field])
            if params[field] < 1:
                raise ValueError(f"priors {field} must be >= 1, got {params[field]!r}")
        if (
            params["burst_lr"] <= 0
            or params["canary_gate_delta"] < 0
            or params["patience_min_gain"] < 0
        ):
            raise ValueError(f"priors file {r.priors_path!r} has out-of-range values")
    return params, from_priors


class CoMitigationPolicy:
    """D: saliency-ordered, budget-paced remap -> BN recalibration -> canary gate ->
    adaptive burst. Per wave k:

    1. Remap. Order from config (D default: Fisher saliency). Greedy = no budget. Paced:
       per-layer budget = floor(available_capacity_layer * rho_k), rho_k = 1/max(1,
       expected_waves - k): the share of what is left this wave would take if the
       remaining waves were equal. expected_waves is the MODELED wear-out schedule the
       controller is told (config.expected_waves, else len(wave_densities)); S2 tests
       robustness when it is wrong. The last expected wave, or any wave past it, gets
       rho=1 (nothing left to reserve for). Pools are per layer, so each layer gets
       its own budget from its own pool.
    2. BN recalibration (forward-only; bn_recal.py) on burst-loader batches, then the
       canary is read: canary_after_recal.
    3. Gate: reference = canary after the PREVIOUS wave's mitigation (the deploy
       baseline for wave 0), so the drop is "damage this wave did that mitigation has
       not already absorbed". drop <= delta => no burst. The health metric is a
       synthetic stress quantity and is never consulted.
    4. Burst: adaptive = PatienceStopper (window-mean plateau rule, capped by t_max),
       fixed = burst_batches. Fresh SGD at burst_lr; Fisher keeps updating
       (system.fisher is ctx.fisher), so later waves rank cells by current gradients.

    Needs reference_mode='living' (config-validated): that is what makes the spares the
    remap allocates trainable by the burst.
    """

    def __init__(self, config: ExperimentConfig):
        # Loaded at construction so a missing/invalid priors file fails before any wave.
        self.params, self.from_priors = effective_params(config)

    def _rho(self, config: ExperimentConfig, wave_index: int) -> float:
        if self.params["remap_pacing"] == "greedy":
            return 1.0
        expected = config.retraining.expected_waves or len(config.fault.wave_densities or [])
        return 1.0 / max(1, expected - wave_index)

    def on_wave(self, ctx: WaveContext) -> dict:
        cfg, system, r, p = ctx.config, ctx.system, ctx.config.retraining, self.params
        paced = p["remap_pacing"] == "paced"
        rho = self._rho(cfg, ctx.wave_index)

        available = {n: int(pool.available_capacity) for n, pool in system.pools.items()}
        budget = (
            {n: math.floor(a * rho) for n, a in available.items()} if paced else None
        )  # greedy: no cap (rho is 1.0 by definition)
        loc = _localize(system)
        remapped_by_layer = system.remap_now(order=r.remap_order, budget=budget)
        unremapped = sum(system.unremapped_effective_stuck().values())
        believed = sum(system.unremapped_decision_stuck().values())

        bn_used = recalibrate_bn(
            system.model, ctx.canary.burst_loader, r.bn_recal_batches, ctx.device
        )
        canary_recal = evaluate_accuracy(system.model, ctx.canary.canary_loader, ctx.device)

        if ctx.gate_reference is not None:
            reference = ctx.gate_reference
        elif ctx.prior_events:
            reference = ctx.prior_events[-1]["canary_acc_after"]
        elif ctx.baseline_canary_acc is not None:
            reference = ctx.baseline_canary_acc
        else:
            raise ValueError("co_mitigation wave 0 needs WaveContext.baseline_canary_acc")
        if r.canary_gate_reference == "best" and ctx.gate_reference is None:
            seen = [e["canary_acc_after"] for e in ctx.prior_events]
            if ctx.baseline_canary_acc is not None:
                seen.append(ctx.baseline_canary_acc)
            reference = max(seen + [reference])
        drop = reference - canary_recal
        # D2 switch gate_noise_k: threshold = k standard errors of the reference reading.
        gate_fields: dict = {}
        if r.gate_noise_k is not None:
            sigma_ref = canary_sigma(reference, r.canary_size)
            threshold = r.gate_noise_k * sigma_ref
            gate_fields = {"gate_threshold": threshold, "canary_sigma": sigma_ref}
        else:
            threshold = p["canary_gate_delta"]
        skipped = drop <= threshold

        fields = {
            "burst_batches": 0,
            "optimizer_steps": 0,
            "stop_reason": "no_burst",
            "canary_trace": [],
            "remapped_this_wave": sum(remapped_by_layer.values()),
            "remapped_by_layer": remapped_by_layer,
            "unremapped_effective_stuck": unremapped,
            **_loc_fields(loc, believed),
            "rho": rho,
            "available_capacity_by_layer": available,
            "budget_by_layer": budget,
            "bn_recal_batches_used": bn_used,
            "canary_after_recal": canary_recal,
            "gate_reference": reference,
            "gate_drop": drop,
            "burst_skipped": skipped,
            **gate_fields,
            "effective_params": {**p, "burst_mode": r.burst_mode, "burst_batches": r.burst_batches},
            "params_from_priors": list(self.from_priors),
        }
        if skipped:
            return fields

        if r.burst_mode == "adaptive":
            min_gain = p["patience_min_gain"]
            if r.noise_aware_plateau:
                # D2 switch: min gain = standard error of the window mean of the canary.
                sigma_recal = canary_sigma(canary_recal, r.canary_size)
                min_gain = sigma_recal / math.sqrt(p["patience_window"])
                fields["canary_sigma_at_recal"] = sigma_recal
                fields["plateau_min_gain"] = min_gain
            kwargs = {
                "stopper": PatienceStopper(
                    p["patience_eval_every"],
                    p["patience_window"],
                    min_gain,
                    p["burst_t_max"],
                ),
                "canary": ctx.canary.canary_loader,
            }
        else:
            kwargs = {
                "n_batches": r.burst_batches,
                "canary": ctx.canary.canary_loader,
                "eval_every": p["patience_eval_every"],
            }
        result = run_burst(
            system,
            ctx.canary.burst_loader,
            lr=p["burst_lr"],
            fisher=ctx.fisher,
            config=cfg,
            device=ctx.device,
            **kwargs,
        )
        fields.update(
            burst_batches=result.batches,
            optimizer_steps=result.optimizer_steps,
            stop_reason=result.stop_reason,
            canary_trace=result.canary_trace,
        )
        if r.restore_best:
            fields.update(restored_best=result.restored_best, best_canary=result.best_canary)
        return fields


class CapacityOrCanaryPolicy(_BasePolicy):
    """C2: C' (burst when spares could not absorb the wave) that ALSO bursts when the canary
    dropped, after remap, by more than gate_noise_k standard errors below the BEST canary
    seen so far (deploy baseline and every earlier wave's canary_acc_after). Fixed-length
    bursts, no BN recalibration, no adaptive stop."""

    def on_wave(self, ctx: WaveContext) -> dict:
        cfg, system, r = ctx.config, ctx.system, ctx.config.retraining
        loc = _localize(system)
        remapped_by_layer: dict[str, int] = {}
        if cfg.mitigation.enable_remapping:
            remapped_by_layer = system.remap_now(order=r.remap_order)
        unremapped = sum(system.unremapped_effective_stuck().values())
        believed = sum(system.unremapped_decision_stuck().values())

        canary_after = evaluate_accuracy(system.model, ctx.canary.canary_loader, ctx.device)
        if ctx.gate_reference is not None:
            reference = ctx.gate_reference
        else:
            seen = [e["canary_acc_after"] for e in ctx.prior_events]
            if ctx.baseline_canary_acc is not None:
                seen.append(ctx.baseline_canary_acc)
            if not seen:
                raise ValueError("capacity_or_canary needs WaveContext.baseline_canary_acc")
            reference = max(seen)
        drop = reference - canary_after
        sigma_ref = canary_sigma(reference, r.canary_size)
        threshold = r.gate_noise_k * sigma_ref
        canary_trigger = drop > threshold
        burst = believed > 0 or canary_trigger

        fields = {
            "burst_batches": 0,
            "optimizer_steps": 0,
            "stop_reason": "no_burst",
            "canary_trace": [],
            "remapped_this_wave": sum(remapped_by_layer.values()),
            "remapped_by_layer": remapped_by_layer,
            "unremapped_effective_stuck": unremapped,
            **_loc_fields(loc, believed),
            "canary_after_recal": canary_after,  # no BN recalibration here: the post-remap canary
            "gate_reference": reference,
            "gate_drop": drop,
            "gate_threshold": threshold,
            "canary_sigma": sigma_ref,
            "burst_skipped": not burst,
            "capacity_trigger": believed > 0,
            "canary_trigger": canary_trigger,
        }
        if burst:
            result = run_burst(
                system,
                ctx.canary.burst_loader,
                lr=r.burst_lr,
                n_batches=r.burst_batches,
                canary=ctx.canary.canary_loader,
                eval_every=r.patience_eval_every,
                fisher=ctx.fisher,
                config=cfg,
                device=ctx.device,
            )
            fields.update(
                burst_batches=result.batches,
                optimizer_steps=result.optimizer_steps,
                stop_reason=result.stop_reason,
                canary_trace=result.canary_trace,
            )
        return fields


class LiuOneShotPolicy:
    """E1: Liu et al. (DAC'17) one-shot rescue, adapted to the wave regime.

    Liu's flow is static: against a known defect map, retrain ONCE with the stuck weights
    frozen, then remap only the most significant defects to a capped redundant pool. No
    pacing, no repeated bursts, no gate, no BN recalibration. Adaptation: every wave remaps
    greedily by significance (config remap_order, Fisher saliency) until the pool is empty
    (no budget), and exactly one fixed-length burst runs in the whole lifetime, at the
    `liu_retrain_at` wave ("first" or "last"; "last" = the wave after which the defect map
    is complete, i.e. the static map Liu assumes). Order within the retrain wave follows
    `liu_remap_after_retrain` (True = Liu's evaluated order: retrain, then remap). The burst
    length (burst_batches) and rate (burst_lr) come from the config; the compute-matched
    run sets burst_batches to D's total steps for the seed."""

    def on_wave(self, ctx: WaveContext) -> dict:
        cfg, system, r = ctx.config, ctx.system, ctx.config.retraining
        n_waves = r.expected_waves or len(cfg.fault.wave_densities or [])
        retrain_wave = 0 if r.liu_retrain_at == "first" else max(0, n_waves - 1)
        retrain_now = ctx.wave_index == retrain_wave
        loc = _localize(system)
        fields: dict = {
            "burst_batches": 0,
            "optimizer_steps": 0,
            "stop_reason": "no_burst",
            "canary_trace": [],
        }

        def burst() -> None:
            result = run_burst(
                system,
                ctx.canary.burst_loader,
                lr=r.burst_lr,
                n_batches=r.burst_batches,
                canary=ctx.canary.canary_loader,
                eval_every=r.patience_eval_every,
                fisher=ctx.fisher,
                config=cfg,
                device=ctx.device,
            )
            fields.update(
                burst_batches=result.batches,
                optimizer_steps=result.optimizer_steps,
                stop_reason=result.stop_reason,
                canary_trace=result.canary_trace,
            )

        if retrain_now and r.liu_remap_after_retrain:
            burst()
        remapped_by_layer = system.remap_now(order=r.remap_order)
        if retrain_now and not r.liu_remap_after_retrain:
            burst()
        unremapped = sum(system.unremapped_effective_stuck().values())
        believed = sum(system.unremapped_decision_stuck().values())
        fields.update(
            remapped_this_wave=sum(remapped_by_layer.values()),
            remapped_by_layer=remapped_by_layer,
            unremapped_effective_stuck=unremapped,
            liu_retrain_wave=retrain_now,
            **_loc_fields(loc, believed),
        )
        return fields


class BnRecalOnlyPolicy:
    """E2: Jung et al. (DATE'21) style forward-only repair. Remap (config order, greedy),
    then refit the BatchNorm running statistics on canary-split training batches
    (retraining.bn_recal_batches). No optimizer, no gradient step, no gate, no burst."""

    def on_wave(self, ctx: WaveContext) -> dict:
        system, r = ctx.system, ctx.config.retraining
        loc = _localize(system)
        remapped_by_layer = system.remap_now(order=r.remap_order)
        unremapped = sum(system.unremapped_effective_stuck().values())
        believed = sum(system.unremapped_decision_stuck().values())
        bn_used = recalibrate_bn(
            system.model, ctx.canary.burst_loader, r.bn_recal_batches, ctx.device
        )
        canary_recal = evaluate_accuracy(system.model, ctx.canary.canary_loader, ctx.device)
        return {
            "burst_batches": 0,
            "optimizer_steps": 0,
            "stop_reason": "no_burst",
            "canary_trace": [],
            "remapped_this_wave": sum(remapped_by_layer.values()),
            "remapped_by_layer": remapped_by_layer,
            "unremapped_effective_stuck": unremapped,
            "bn_recal_batches_used": bn_used,
            "canary_after_recal": canary_recal,
            **_loc_fields(loc, believed),
        }


_POLICIES = {
    "none": NonePolicy,
    "static": StaticPolicy,
    "periodic": PeriodicPolicy,
    "capacity": CapacityPolicy,
    "co_mitigation": CoMitigationPolicy,
    "capacity_or_canary": CapacityOrCanaryPolicy,
    "liu_oneshot": LiuOneShotPolicy,
    "bn_recal_only": BnRecalOnlyPolicy,
}


def build_policy_for(config: ExperimentConfig):
    cls = _POLICIES[config.retraining.policy]
    return cls(config) if cls is CoMitigationPolicy else cls()
