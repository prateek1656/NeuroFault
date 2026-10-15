"""Typed experiment configuration. Everything downstream takes one of these
dataclasses instead of raw dicts/kwargs, so required fields (like the
enable_* mitigation flags) can't be silently omitted or ignored."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class DeviceConfig:
    name: str = "vteam"  # key into devices/registry.py
    r_on: float = 100.0
    r_off: float = 10000.0
    time_series_resolution: float = 1e-10
    extra_params: dict = field(default_factory=dict)  # device-specific overrides

    def __post_init__(self):
        if not (self.r_on > 0 and self.r_off > 0):
            raise ValueError(
                f"DeviceConfig.r_on/r_off must be positive (real resistances), "
                f"got r_on={self.r_on!r}, r_off={self.r_off!r}"
            )
        if self.r_on >= self.r_off:
            raise ValueError(
                f"DeviceConfig.r_on must be less than r_off (on-resistance < "
                f"off-resistance, real device physics), got r_on={self.r_on!r}, "
                f"r_off={self.r_off!r}"
            )
        if self.time_series_resolution <= 0:
            raise ValueError(
                f"DeviceConfig.time_series_resolution must be positive, "
                f"got {self.time_series_resolution!r}"
            )


@dataclass
class CrossbarConfig:
    scheme: str = "DoubleColumn"  # "DoubleColumn" | "SingleColumn"
    tile_shape: tuple[int, int] | None = None
    patch_layer_types: list[str] = field(default_factory=lambda: ["Linear", "Conv2d"])
    max_input_voltage: float = 0.3
    adc_resolution: int = 8  # ADC bits - quantizes the read-out current/output
    dac_resolution: int = 8  # DAC bits - quantizes the input voltages
    redundancy_factor: float = 0.15
    # "idealized_per_cell": remapping restores a faulty cell's own reference
    #   conductance in place - no physical spare-hardware analogue (Phase 1 design decision).
    # "row_col_granularity": repairs whole rows/columns at once (word-line/
    #   bit-line resolution, matching how real spare hardware is organized) -
    #   see mitigation/remap.py and crossbar/self_healing.py.
    redundancy_scheme: str = "idealized_per_cell"
    # row_col_granularity only: fraction of a row's/column's own cells that
    # must be critical before that row/column is condemned and restored -
    # condemning on a single critical cell would waste an entire row/column's
    # worth of scarce spare capacity on one bad cell.
    redundancy_condemn_threshold: float = 0.5

    # CrossSim-only (crosssim_backend.py's _build_params): sets
    # params.simulation.useGPU, a real, existing CrossSim mechanism (see
    # simulator/parameters/simulation_parameters.py and backend/backend.py's
    # ComputeBackend singleton, which transparently swaps its whole array
    # module between numpy and cupy) - not something this project built.
    # Default False so every existing config/result on this CPU-only Mac is
    # unaffected. Requires the "gpu" extra (cupy) installed and actual CUDA
    # hardware - verify on the real rented box before trusting a long run,
    # per this project's own verify-before-committing-compute norm; this
    # development Mac has no CUDA to test the path against.
    use_gpu: bool = False

    # Circuit-level access topology.
    # None (default) preserves today's fully-idealized behavior (no parasitics/
    # series resistance/current compliance modeled) - every existing experiment
    # config's already-verified numbers stay unchanged unless this is set
    # explicitly. "1T1R" (transistor per cell - near-ideal isolation) | "0T1R"
    # (bare crossbar - worst-case sneak path) | "1S1R" (selector device in
    # series) - see crosssim_backend.py/aihwkit_backend.py for the real,
    # physically-meaningful knobs (parasitic wire/terminal resistance, current
    # compliance, series resistance, IR drop) each value maps to.
    circuit_topology: str | None = None

    # Devices combined per synapse ("multi-device-per-synapse / bit-slicing" goal). 1 (default) =
    # today's single-device-per-synapse
    # behavior, unchanged. >1: CrossSim uses its native bit-sliced core
    # (core.style=BITSLICED); AIHWKit wraps the device in a VectorUnitCell of
    # that many physical devices. Orthogonal to `scheme` (DoubleColumn's
    # signed-weight column split), same as tile_shape/redundancy_scheme are.
    devices_per_synapse: int = 1

    # CrossSim-only (crosssim_backend.py's calibrate_adc_ranges()/patch_layers()):
    # False preserves today's ADCRangeLimits.MAX behavior exactly, unchanged. Real
    # bug found and fixed during development: MAX sets the ADC's
    # quantization range to a theoretical worst-case that scales LINEARLY with
    # crossbar size, while real dot-product signal only scales ~sqrt(size) -
    # wastes most of the ADC's resolution on large crossbars specifically.
    # True=run a real calibration pass (adc_calibration_images real samples
    # through the model) before patching, deriving each layer's ADC range from
    # its own actual internal signal (via CrossSim's profile_adc_inputs
    # instrumentation - NOT the layer's raw PyTorch output, a different numeric
    # space entirely, the bug behind two earlier failed attempts) instead of the
    # worst-case bound. Verified for real: closes a real, if modest gap (+0.42pp
    # accuracy on a 400-batch local comparison, 10/10 layers' signal improved).
    calibrate_adc: bool = False
    # Real samples used for the calibration pass above. 320 (10 batches at
    # batch_size=32) is what the development-time verification used - large enough
    # that the observed range's growth from half- to full-sample was consistently
    # small (a couple percent, not double-digit), the actual empirical
    # justification, not a round-number guess.
    adc_calibration_images: int = 320

    # CrossSim-only (crosssim_backend.py's _build_params()): False preserves today's
    # behavior exactly, unchanged. Real, SEPARATE bug found and fixed after
    # calibrate_adc: calibrate_adc only fixes the ADC's (output-side)
    # quantization range. The DAC (input side - digitizes each layer's real activations
    # before the analog multiply) was left at CrossSim's own hardcoded default clip range
    # of [-1, 1], completely independent of calibrate_adc and never touched by it. Real
    # activations routinely exceed this (e.g. a ReLU'd conv activation reaching ~5.6) -
    # verified directly by instrumenting the actual DAC call: 7.57% of one layer's real
    # input values were silently hard-clipped to 1.0 on every single forward pass,
    # regardless of adc_resolution/dac_resolution bit depth (a fixed clip range isn't
    # fixed by more precision - this is exactly why an earlier bit-depth sweep
    # "saturated": the bottleneck was never resolution). True enables CrossSim's own
    # native per-call input-percentile-scaling (params.core.mapping.inputs.mvm/vmm.
    # percentile), which recomputes each layer's real input range from that exact
    # forward call's actual data every time - unlike calibrate_adc, this needs no
    # separate calibration pass or calibration_images, since it can never go stale
    # relative to the batch it's scaling. Verified for real: isolated single-layer
    # cosine similarity 0.9723->0.9997 (8-bit), full 9-layer network logits cosine
    # 0.6252->0.9839 (8-bit) / ->1.0000 (16-bit), and eval accuracy on a real pretrained
    # model's held-out batch went from 0.156 to 0.500 - exactly matching the float
    # model's own accuracy on that same batch.
    calibrate_dac: bool = False

    def __post_init__(self):
        if self.adc_resolution < 1 or self.dac_resolution < 1:
            raise ValueError(
                f"CrossbarConfig.adc_resolution/dac_resolution must be >= 1 bit, got "
                f"adc_resolution={self.adc_resolution!r}, dac_resolution={self.dac_resolution!r}"
            )
        if not (0.0 <= self.redundancy_factor <= 1.0):
            raise ValueError(
                f"CrossbarConfig.redundancy_factor must be in [0, 1] (a fraction of "
                f"cells/rows/cols), got {self.redundancy_factor!r}"
            )
        if not (0.0 <= self.redundancy_condemn_threshold <= 1.0):
            raise ValueError(
                "CrossbarConfig.redundancy_condemn_threshold must be in [0, 1] (a "
                f"fraction of a row's/column's cells), got {self.redundancy_condemn_threshold!r}"
            )
        if self.devices_per_synapse < 1:
            raise ValueError(
                f"CrossbarConfig.devices_per_synapse must be >= 1, got {self.devices_per_synapse!r}"
            )
        if self.adc_calibration_images < 1:
            raise ValueError(
                "CrossbarConfig.adc_calibration_images must be >= 1, got "
                f"{self.adc_calibration_images!r}"
            )
        if self.tile_shape is not None and (self.tile_shape[0] <= 0 or self.tile_shape[1] <= 0):
            raise ValueError(
                f"CrossbarConfig.tile_shape dimensions must be positive, got {self.tile_shape!r}"
            )
        if self.circuit_topology is not None and self.circuit_topology not in (
            "1T1R",
            "0T1R",
            "1S1R",
        ):
            raise ValueError(
                f"Unknown crossbar_config.circuit_topology: {self.circuit_topology!r} "
                "(expected None, '1T1R', '0T1R', or '1S1R')"
            )
        if self.redundancy_scheme not in ("idealized_per_cell", "row_col_granularity"):
            raise ValueError(
                f"Unknown crossbar_config.redundancy_scheme: {self.redundancy_scheme!r} "
                "(expected 'idealized_per_cell' or 'row_col_granularity')"
            )


@dataclass
class FaultConfig:
    density: float = 0.1
    distribution: str = "random"  # "random" | "clustered" | "gradient" | "line" - spatial pattern,
    #   only used by failure_model="density"; orthogonal to failure_model (see below).
    injection_interval: int = 8  # batches between re-injections

    # failure_model picks WHICH cells fail and HOW MANY, each injection call:
    # "density": today's behavior - a flat density * total_cells picked fresh
    #   each call via `distribution`, no memory of prior calls. Hard stuck-at
    #   (pinned to g_min/g_max) - permanent, never recoverable.
    # "weibull" | "exponential": real time-to-failure models - each cell's
    #   own accumulated simulated cycles (HealthMonitor.cycles) determines its
    #   own conditional failure probability this interval. See
    #   faults/reliability.py and faults/injection.py::_statistical_stuck_at.
    #   Also hard stuck-at, just a different arrival-time distribution.
    # "drift": a real, distinct failure mode found missing during development - soft_mitigation was
    # built to
    #   correct "drift/variability, not a hardware-stuck-at device" (see its
    #   own docstring) but this project never actually injected drift, only
    #   stuck-at, so that strategy was never tested against the failure mode
    #   it targets. Cells shift gradually toward one conductance rail
    #   (drift_direction) by drift_magnitude*g_range each injection call,
    #   accumulating over repeated calls (clamped to [g_min, g_max]) - never
    #   marked stuck_mask, since the cell stays writable/functional, just
    #   imprecise. See drift_magnitude/drift_direction below.
    failure_model: str = "density"
    weibull_shape: float = 2.0  # dimensionless shape (k) - 2.0 is the classic
    #   wear-out region (increasing hazard rate), a modeling choice not a
    #   device measurement, so it's the one statistical parameter with a
    #   real default.
    weibull_scale: float | None = None  # characteristic life in simulated
    #   cycles (lambda) - REQUIRED when failure_model="weibull"; no invented
    #   default, must be a real value for the device being modeled.
    exponential_rate: float | None = None  # REQUIRED when
    #   failure_model="exponential"; no invented default.
    # Arrhenius is a modifier on weibull/exponential, not a fourth model -
    # None (default) means no temperature acceleration at all.
    arrhenius_activation_energy_ev: float | None = None
    temperature_kelvin: float = 300.0  # ~27C, room temperature
    reference_temperature_kelvin: float = 300.0  # the temperature
    #   weibull_scale/exponential_rate were characterized at; acceleration
    #   factor is exactly 1.0 when temperature_kelvin == this.

    # failure_model="drift" only. No single canonical value exists in the
    # literature the way stuck-at density figures do (checked during development -
    # retention/endurance papers describe the mechanism, not a portable
    # per-cycle magnitude) - this is a documented modeling judgment call, not
    # a lookup. 0.05 (5% of g_range per injection event) is a starting point
    # chosen so repeated injections (this project's own moderate-regime
    # calibration uses ~3 events) compound into a meaningfully degraded but
    # not instantly rail-clamped cell.
    drift_magnitude: float = 0.05
    # "toward_hrs": literature on oxide RRAM retention loss most commonly
    #   describes LRS filaments as the less stable state, spontaneously
    #   losing conductance (drifting toward HRS/g_min) via oxygen-vacancy
    #   diffusion - a real but not universal-across-all-technologies
    #   direction, used as the default with this caveat stated plainly.
    # "toward_lrs": the opposite bias, for devices/technologies where that's
    #   the better-supported direction - a config choice, not a default.
    # "random_per_cell": each affected cell's direction is decided once (at
    #   its first drift event) and reused for every subsequent event -
    #   matches real drift being monotonic per-device, not a fault model
    #   that flips direction call to call.
    drift_direction: str = "toward_hrs"

    # failure_model="density" only. Fraction of newly stuck cells that stick at
    # LRS (high conductance, g_max); the rest stick at HRS (low conductance,
    # g_min). 0.5 is the old fixed 50:50 split (default, so existing runs are
    # unchanged). Measured RRAM arrays report mostly HRS failures (9.04% HRS : 1.75% LRS).
    stuck_lrs_fraction: float = 0.5

    # Deployment regime only (ExperimentConfig.regime="deployment"): one fault
    # density per wave EVENT, in order - the lifetime is exactly
    # len(wave_densities) injection events, each driven by run_deployment()
    # rather than by `injection_interval`'s batch cadence (there is no training
    # loop to count batches in). Required (non-empty) in the deployment
    # regime unless the config only trains the pre-deploy checkpoint, and
    # FORBIDDEN in the continuous regime - silently ignoring a schedule the
    # caller wrote down is how a run ends up measuring something other than
    # what its config says. Applies to the primary `fault` channel with
    # failure_model="density" (the only model with a per-call density).
    wave_densities: list[float] | None = None

    def __post_init__(self):
        if self.wave_densities is not None:
            if len(self.wave_densities) == 0:
                raise ValueError(
                    "FaultConfig.wave_densities must be non-empty when set (use None for "
                    "'no wave schedule'), got []"
                )
            for d in self.wave_densities:
                if not (0.0 < d <= 1.0):
                    raise ValueError(
                        "FaultConfig.wave_densities entries must each be in (0, 1] (a wave "
                        f"that sticks no cells is not a wave), got {self.wave_densities!r}"
                    )
        if self.failure_model == "weibull" and self.weibull_scale is None:
            raise ValueError(
                "FaultConfig.failure_model='weibull' requires weibull_scale to be set "
                "explicitly - no invented default for a device's characteristic life."
            )
        if self.failure_model == "exponential" and self.exponential_rate is None:
            raise ValueError(
                "FaultConfig.failure_model='exponential' requires exponential_rate to be "
                "set explicitly - no invented default for a device's failure rate."
            )
        if self.failure_model not in ("density", "weibull", "exponential", "drift"):
            raise ValueError(
                f"Unknown FaultConfig.failure_model: {self.failure_model!r} "
                "(expected 'density', 'weibull', 'exponential', or 'drift')"
            )
        if self.drift_direction not in ("toward_hrs", "toward_lrs", "random_per_cell"):
            raise ValueError(
                f"Unknown FaultConfig.drift_direction: {self.drift_direction!r} "
                "(expected 'toward_hrs', 'toward_lrs', or 'random_per_cell')"
            )
        if not (0.0 <= self.stuck_lrs_fraction <= 1.0):
            raise ValueError(
                f"FaultConfig.stuck_lrs_fraction must be in [0, 1], got {self.stuck_lrs_fraction!r}"
            )
        if not (0.0 <= self.drift_magnitude <= 1.0):
            raise ValueError(
                f"FaultConfig.drift_magnitude must be in [0, 1], got {self.drift_magnitude!r}"
            )
        if self.distribution not in ("random", "clustered", "gradient", "line"):
            raise ValueError(
                f"Unknown FaultConfig.distribution: {self.distribution!r} "
                "(expected 'random', 'clustered', 'gradient', or 'line')"
            )
        if not (0.0 <= self.density <= 1.0):
            raise ValueError(f"FaultConfig.density must be in [0, 1], got {self.density!r}")
        if self.injection_interval < 1:
            raise ValueError(
                f"FaultConfig.injection_interval must be >= 1, got {self.injection_interval!r}"
            )
        if self.weibull_shape <= 0:
            raise ValueError(
                f"FaultConfig.weibull_shape must be positive, got {self.weibull_shape!r}"
            )
        if self.weibull_scale is not None and self.weibull_scale <= 0:
            raise ValueError(
                f"FaultConfig.weibull_scale must be positive, got {self.weibull_scale!r}"
            )
        if self.exponential_rate is not None and self.exponential_rate <= 0:
            raise ValueError(
                f"FaultConfig.exponential_rate must be positive, got {self.exponential_rate!r}"
            )
        if self.temperature_kelvin <= 0 or self.reference_temperature_kelvin <= 0:
            raise ValueError(
                "FaultConfig.temperature_kelvin/reference_temperature_kelvin must be "
                f"positive (real Kelvin), got temperature_kelvin={self.temperature_kelvin!r}, "
                f"reference_temperature_kelvin={self.reference_temperature_kelvin!r}"
            )


@dataclass
class MitigationConfig:
    enable_soft_mitigation: bool = False
    enable_remapping: bool = False
    enable_layer_reset: bool = False
    enable_predictive: bool = True
    force_remapping: bool = False
    health_threshold: float = 75.0
    health_check_interval: int = 10  # batches between check_health() calls
    # "full_diff" (default): read every cell, compare to reference - always
    #   correct, models the expensive naive approach real hardware avoids.
    # "checksum": cheap row/column-sum first pass (see health_monitor.py);
    #   falls back to full_diff automatically once a mismatch is found, so
    #   fault localization is identical whenever a fault actually exists -
    #   only the no-fault case gets cheaper (in modeled cell-touch terms).
    detection_method: str = "full_diff"
    # checksum only: real detection-method upgrade made during development - replaces a
    # fixed-fraction-of-
    # g_range threshold with a variance-based adaptive one (V-ABFT's
    # approach, 2026: derive the flagging threshold from the OBSERVED spread
    # of row/column checksum deviations across the whole crossbar this call,
    # rather than assuming a fixed magnitude always means "fault"). A
    # row/column is flagged when its deviation exceeds
    # mean(deviation) + checksum_z_threshold * std(deviation) - the classic
    # z-score outlier convention (3.0 = "3-sigma"). Honest limitation, same
    # as the fixed-threshold version it replaces: self-referential to the
    # current population, so it degrades once faults are a majority rather
    # than a minority of cells (the population's own mean/std become
    # dominated by the faults) - see health_monitor.py.
    checksum_z_threshold: float = 3.0
    # Real design flaw found and fixed during development:
    # layer_reset wipes an ENTIRE layer's weights back to their random
    # pre-training initialization - unlike soft_mitigation (a small nudge)
    # and remapping (capacity-limited to redundancy_factor), it had no
    # throttle at all, so once avg_health crossed health_threshold it fired
    # on essentially every subsequent health check for the rest of training
    # (measured: 222 times over 1257 batches in a real run), repeatedly
    # discarding legitimate learning - actively worse than no mitigation at
    # all. This gates it to at most once every N check_health() CALLS
    # (project-wide, not per-batch, so it composes with health_check_interval
    # rather than introducing an unrelated batch-count knob) per layer -
    # "last resort" should mean rare, not near-constant.
    layer_reset_cooldown_checks: int = 10

    def __post_init__(self):
        if not (0.0 <= self.health_threshold <= 100.0):
            raise ValueError(
                f"MitigationConfig.health_threshold must be in [0, 100], got {self.health_threshold!r}"
            )
        if self.health_check_interval < 1:
            raise ValueError(
                f"MitigationConfig.health_check_interval must be >= 1, got {self.health_check_interval!r}"
            )
        if self.detection_method not in ("full_diff", "checksum"):
            raise ValueError(
                f"Unknown MitigationConfig.detection_method: {self.detection_method!r} "
                "(expected 'full_diff' or 'checksum')"
            )
        if self.checksum_z_threshold < 0:
            raise ValueError(
                "MitigationConfig.checksum_z_threshold must be >= 0, got "
                f"{self.checksum_z_threshold!r}"
            )
        if self.layer_reset_cooldown_checks < 0:
            raise ValueError(
                "MitigationConfig.layer_reset_cooldown_checks must be >= 0, got "
                f"{self.layer_reset_cooldown_checks!r}"
            )


_RETRAINING_POLICIES = (
    "none",
    "static",
    "periodic",
    "capacity",
    "co_mitigation",
    "capacity_or_canary",
    "liu_oneshot",
    "bn_recal_only",
)
_LIU_RETRAIN_AT = ("first", "last")
_REFERENCE_MODES = ("frozen", "living", "living_orig")
_REMAP_ORDERS = ("prefix", "random", "saliency", "magnitude")
_REMAP_PACINGS = ("greedy", "paced")
_BURST_MODES = ("fixed", "adaptive")


@dataclass
class RetrainingConfig:
    # Fault-aware retraining baseline: instead of runtime mitigation reacting to faults during
    # inference,
    # this bakes a fixed, known defect map into the crossbar once before
    # training and never changes it again - gradient descent itself learns
    # weights for the surviving cells that compensate for the permanently
    # stuck ones. When enabled, ExperimentConfig.fault still controls the
    # defect map's density/distribution/failure_model (same generation
    # mechanism, just triggered once instead of periodically), and
    # mitigation.enable_* should be set to False since retraining itself is
    # the compensation mechanism here.
    #
    # LEGACY flag, continuous regime only: kept so every existing config and
    # result keeps its exact meaning. New work uses `policy` below; setting both
    # is rejected (ExperimentConfig.__post_init__) rather than guessing which wins.
    enable_fault_aware: bool = False

    # --- Co-mitigation controller ---------------
    # WHEN retraining/structural mitigation is triggered against the wear-out curve:
    # "none": never (the R0/R1 arms; R1 = remapping alone).
    # "static": one burst after the first wave.   "periodic": a burst after every wave.
    # "capacity": a burst only when a wave exceeds remap capacity.
    # "co_mitigation": the controller (canary-gated, saliency-ordered, paced remap + BN
    #   recalibration + adaptive burst). All non-"none" policies are implemented by
    #   neurofault/retraining (see policy_hook.py); config validation only fixes the
    #   vocabulary and cross-field constraints.
    policy: str = "none"
    # Burst optimizer LR. ONE shared value from the calibration sweep, used by every
    # burst arm, so no arm gets an LR-tuning advantage over another.
    burst_lr: float = 1e-3
    # Fixed-length burst (static/periodic/capacity arms), in training batches.
    burst_batches: int = 200
    # Adaptive-burst hard cap (co_mitigation); the patience rule below usually stops sooner.
    burst_t_max: int = 1000
    # Patience stopper: every `patience_eval_every` batches evaluate the canary; stop when
    # the moving-average gain over the last `patience_window` evals is < patience_min_gain
    # (accuracy points) or burst_t_max is hit. A plateau rule, not an online curve fit:
    # canary SE is ~1pp, so epsilon-gain stops on a fitted curve would be noise-driven.
    patience_window: int = 3
    patience_eval_every: int = 25
    patience_min_gain: float = 0.5
    # Canary gate: skip a burst when the canary accuracy drop vs the pre-wave canary is
    # <= this many accuracy points.
    canary_gate_delta: float = 1.0
    # Gate reference: 'previous' = canary after the previous wave's mitigation (original);
    # 'best' = best canary seen so far incl. the deploy baseline (stops cumulative erosion
    # over many small waves).
    canary_gate_reference: str = "previous"
    # Held-out train-index split excluded from burst training and evaluated through the
    # non-augmenting test transform (retraining/canary.py).
    canary_size: int = 1024
    # Forward-only BatchNorm running-stat recalibration batches after a wave (0 = off).
    bn_recal_batches: int = 8
    # "frozen": a remapped spare is pinned to the reference captured at system
    #   construction. "living": the spare takes the CURRENT weight and trains normally
    #   (needs per-cell spares: crossbar.redundancy_scheme="idealized_per_cell").
    reference_mode: str = "frozen"
    # Stuck-weight freeze (system.py): snapshot each newly-stuck cell's .weight and restore it
    # after every synchronize(), so grad/momentum/weight-decay drift can't leave garbage that a
    # later remap copies into a spare. None = auto: on in the deployment regime, off in the
    # legacy continuous regime. Off keeps the continuous path byte-identical to items 1-3's
    # results: the freeze is not strictly neutral (CrossSim re-derives array limits from the
    # whole .weight matrix, so a drifting stuck weight ulp-perturbs other cells; measured
    # ~1e-7 relative in grad_norm, nothing else) - see system.py.
    freeze_stuck_weights: bool | None = None
    # Which stuck cells get spares first. "prefix" is the legacy row-major order (kept
    # as the default so existing behavior is unchanged) - a strawman for new arms, since
    # the lowest rows get every spare; new arms use random/saliency/magnitude.
    remap_order: str = "prefix"
    # "greedy": spend capacity as waves arrive. "paced": reserve spares for later waves
    # (budget = remaining capacity / expected remaining waves).
    remap_pacing: str = "greedy"
    # Total wave count the pacer plans for. None = len(fault.wave_densities) in
    # deployment (resolved at run time), required for "paced" in continuous regime.
    expected_waves: int | None = None
    # JSON of calibrated controller parameters
    # (experiments/configs/retrain_controller_priors.json), written by
    # scripts/calibrate_retrain_controller.py from the GPU calibration sweep. None = the
    # fields above are authoritative. When set (co_mitigation only), the file's eta, delta,
    # kappa, k, m, t_max and pacing override burst_lr, canary_gate_delta,
    # patience_min_gain, patience_window, patience_eval_every, burst_t_max and
    # remap_pacing, and each event records which values came from the file. A missing
    # file is an error (run_deployment checks before any compute), never a silent
    # fallback to the config fields: that would run an uncalibrated "D" under D's name.
    priors_path: str | None = None
    # How the co_mitigation burst is sized: "adaptive" = PatienceStopper (plateau of the
    # canary or burst_t_max), "fixed" = burst_batches (the noadapt ablation). Read by
    # co_mitigation only; the other policies always run fixed bursts.
    burst_mode: str = "adaptive"
    # EMA decay of the diagonal-Fisher saliency tracker (retraining/saliency.py).
    fisher_ema_decay: float = 0.99
    # --- D2 switches. All off by default. ---
    # If set, gate threshold = gate_noise_k * canary standard error (replaces canary_gate_delta).
    gate_noise_k: float | None = None
    # Plateau min gain = canary standard error / sqrt(patience_window).
    noise_aware_plateau: bool = False
    # When the plateau rule fires, run anneal_steps more steps with the LR falling to 0.
    anneal_on_stop: bool = False
    # Keep the weights of the best canary reading; restore them if the final reading is lower.
    restore_best: bool = False
    anneal_steps: int = 50
    # --- E1/E2 baselines. Read only by their policies. ---
    # liu_oneshot (Liu et al., DAC'17 adapted to waves): the ONE retraining burst happens at the
    # "first" wave or the "last" wave of the lifetime; every wave remaps greedily (no pacing).
    liu_retrain_at: str = "last"
    # True = Liu's evaluated order (retrain against the stuck map, THEN remap the most significant
    # defects); False = remap first, then retrain.
    liu_remap_after_retrain: bool = True

    def __post_init__(self):
        if self.liu_retrain_at not in _LIU_RETRAIN_AT:
            raise ValueError(
                f"Unknown RetrainingConfig.liu_retrain_at: {self.liu_retrain_at!r} "
                f"(expected one of {_LIU_RETRAIN_AT})"
            )
        if self.gate_noise_k is not None and self.gate_noise_k <= 0:
            raise ValueError(
                f"RetrainingConfig.gate_noise_k must be > 0 or None, got {self.gate_noise_k!r}"
            )
        if self.anneal_steps < 1:
            raise ValueError(
                f"RetrainingConfig.anneal_steps must be >= 1, got {self.anneal_steps!r}"
            )
        if self.policy not in _RETRAINING_POLICIES:
            raise ValueError(
                f"Unknown RetrainingConfig.policy: {self.policy!r} "
                f"(expected one of {_RETRAINING_POLICIES})"
            )
        if self.reference_mode not in _REFERENCE_MODES:
            raise ValueError(
                f"Unknown RetrainingConfig.reference_mode: {self.reference_mode!r} "
                "(expected 'frozen', 'living' or 'living_orig')"
            )
        if self.remap_order not in _REMAP_ORDERS:
            raise ValueError(
                f"Unknown RetrainingConfig.remap_order: {self.remap_order!r} "
                f"(expected one of {_REMAP_ORDERS})"
            )
        if self.remap_pacing not in _REMAP_PACINGS:
            raise ValueError(
                f"Unknown RetrainingConfig.remap_pacing: {self.remap_pacing!r} "
                "(expected 'greedy' or 'paced')"
            )
        if self.burst_mode not in _BURST_MODES:
            raise ValueError(
                f"Unknown RetrainingConfig.burst_mode: {self.burst_mode!r} "
                "(expected 'fixed' or 'adaptive')"
            )
        if self.policy == "co_mitigation":
            if self.reference_mode != "living":
                raise ValueError(
                    "retraining.policy='co_mitigation' requires reference_mode='living': "
                    "the controller's retraining burst can only compensate for unremapped "
                    "cells if the spares it allocates are trainable (frozen spares are "
                    "pinned to the old reference)"
                )
            if self.burst_mode == "fixed" and self.burst_batches < 1:
                raise ValueError(
                    "RetrainingConfig.burst_batches must be >= 1 for co_mitigation with "
                    f"burst_mode='fixed', got {self.burst_batches!r}"
                )
        elif self.priors_path is not None:
            raise ValueError(
                "RetrainingConfig.priors_path is only read by policy='co_mitigation', got "
                f"policy={self.policy!r}"
            )
        if self.burst_lr <= 0:
            raise ValueError(f"RetrainingConfig.burst_lr must be positive, got {self.burst_lr!r}")
        if self.burst_batches < 0:
            raise ValueError(
                f"RetrainingConfig.burst_batches must be >= 0, got {self.burst_batches!r}"
            )
        if self.policy == "capacity_or_canary" and self.gate_noise_k is None:
            raise ValueError("RetrainingConfig.policy='capacity_or_canary' needs gate_noise_k")
        if (
            self.policy in ("static", "periodic", "capacity", "capacity_or_canary", "liu_oneshot")
            and self.burst_batches < 1
        ):
            raise ValueError(
                f"RetrainingConfig.burst_batches must be >= 1 for policy={self.policy!r} "
                f"(a fixed-length burst arm with zero batches never retrains), got "
                f"{self.burst_batches!r}"
            )
        if self.burst_t_max < 1:
            raise ValueError(f"RetrainingConfig.burst_t_max must be >= 1, got {self.burst_t_max!r}")
        if self.patience_window < 1:
            raise ValueError(
                f"RetrainingConfig.patience_window must be >= 1, got {self.patience_window!r}"
            )
        if self.patience_eval_every < 1:
            raise ValueError(
                "RetrainingConfig.patience_eval_every must be >= 1, got "
                f"{self.patience_eval_every!r}"
            )
        if self.patience_min_gain < 0:
            raise ValueError(
                "RetrainingConfig.patience_min_gain must be >= 0 (accuracy points), got "
                f"{self.patience_min_gain!r}"
            )
        if self.canary_gate_reference not in ("previous", "best"):
            raise ValueError(
                f"RetrainingConfig.canary_gate_reference must be 'previous' or 'best', got {self.canary_gate_reference!r}"
            )
        if self.canary_gate_delta < 0:
            raise ValueError(
                "RetrainingConfig.canary_gate_delta must be >= 0 (accuracy points), got "
                f"{self.canary_gate_delta!r}"
            )
        if self.canary_size < 1:
            raise ValueError(f"RetrainingConfig.canary_size must be >= 1, got {self.canary_size!r}")
        if self.bn_recal_batches < 0:
            raise ValueError(
                f"RetrainingConfig.bn_recal_batches must be >= 0, got {self.bn_recal_batches!r}"
            )
        if self.expected_waves is not None and self.expected_waves < 1:
            raise ValueError(
                f"RetrainingConfig.expected_waves must be >= 1 or None, got {self.expected_waves!r}"
            )
        if not (0.0 <= self.fisher_ema_decay < 1.0):
            raise ValueError(
                f"RetrainingConfig.fisher_ema_decay must be in [0, 1), got {self.fisher_ema_decay!r}"
            )


_REGIMES = ("continuous", "deployment")


@dataclass
class ExperimentConfig:
    name: str
    seed: int = 42
    simulator: str = "crosssim"  # "crosssim" | "aihwkit" - which crossbar simulation
    #   backend patches the model.
    dataset: str = "cifar10"  # "cifar10" | "cifar100" | "mnist"
    num_classes: int = 10  # classifier head width; must be 100 for cifar100 (10 otherwise)
    model_arch: str = "resnet9"  # "resnet9" (SimpleCNN) | "vgg6" (VGG6, plain 6-conv) | "vit_tiny" (ViTTiny, 4-block ViT, no BN) | "tiny_cnn" (TinyCNN, ~42K weights -
    #   the fast logic-test path; never used for a reported number).
    train_samples: int = 2000
    test_samples: int = 1000
    batch_size: int = 32
    learning_rate: float = 0.001
    # Real gap found during development:
    # no L2 regularization at all, on a model well below realistic accuracy
    # on a severely-limited-data regime - a real lever for generalization,
    # not architecture. 0.0 (off) preserves prior behavior exactly.
    weight_decay: float = 0.0
    # Cosine-anneals the optimizer's LR from learning_rate down to ~0 over
    # the full run (Loshchilov & Hutter, SGDR) - same investigation's finding:
    # a flat LR the whole run plateaus early once augmentation is in the
    # mix (augmentation needs more effective training time to pay off, and
    # a decaying LR is standard practice for actually using that time to
    # converge rather than oscillate). False preserves prior behavior.
    lr_cosine_schedule: bool = False
    # Additive, opt-in fields for replicating IBM's actual published
    # CIFAR-10/ResNet-9 training recipe (Le Gallo et al. 2023, Nature Electronics - verified against
    # the arXiv preprint's Methods section, not guessed). Defaults preserve every existing
    # config's behavior exactly (Adam, no momentum, no one-cycle, no label
    # smoothing) - only configs that opt in are affected.
    optimizer: str = "adam"  # "adam" | "adamw" | "sgd"
    momentum: float = 0.9  # only used when optimizer="sgd"
    lr_one_cycle: bool = False  # torch.optim.lr_scheduler.OneCycleLR, peak=learning_rate -
    #   mutually exclusive with lr_cosine_schedule (validated below); the actual technique
    #   ("superconvergence", Smith 2018) the canonical ResNet-9/DAWNBench recipe uses to reach
    #   its accuracy in far fewer epochs than plain cosine annealing.
    label_smoothing: float = 0.0  # nn.CrossEntropyLoss(label_smoothing=...), 0.0 preserves
    #   prior behavior exactly.
    grad_clip_norm: float = 0.0  # 0.0 = disabled, preserves prior behavior. Real failure hit
    #   on a rented GPU: SGD + one-cycle LR training loss went to nan (health crashed 92%->0%)
    #   at epoch 2/50, batch ~1100 - clean and healthy the batch before. Classic SGD-divergence
    #   pattern with no gradient clipping; IBM's own paper explicitly clips gradients (max
    #   l2-norm 5.0) for its Adam-trained LSTM phase, and this project's own analog-quantized
    #   forward pass adds noise on top of a clean digital ResNet-9 that IBM's recipe never had
    #   to contend with - real reason to expect *more* instability here, not less.
    float_pretrain_epochs: int = 0  # 0 = disabled, preserves prior behavior exactly (train
    #   end-to-end through the analog/crossbar forward pass from epoch 0, as every config did
    #   before this). When > 0, the first N epochs train the plain (unpatched) float model with
    #   no crossbar simulation at all, and only the *remaining* epochs run through the analog
    #   forward pass - matching IBM's own published hardware-aware training methodology (Rasch
    #   et al. 2023, arXiv:2302.08469, Fig. 1: "DNNs are first trained in 32bit floating-point
    #   (FP32), then re-trained in a hardware-aware manner"), not guessed. Real motivation: a
    #   direct per-layer signal probe measured the analog forward pass collapsing
    #   signal correlation to as low as cosine=0.47 in ResNet-9's deepest residual stage - asking
    #   an optimizer to learn CIFAR-10 from a random init AND push a usable gradient through that
    #   much noise simultaneously is a much harder problem than the literature ever poses.
    finetune_learning_rate: float = 0.0  # 0.0 sentinel = use `learning_rate` for the analog
    #   phase too, preserving prior behavior exactly. Only meaningful when float_pretrain_epochs
    #   > 0. Real failure this field fixes: the first hardware-aware-pretrain run
    #   reused `learning_rate`'s full one-cycle schedule (peak 0.05) for the analog fine-tune
    #   phase - i.e. restarted a from-scratch-scale LR ramp against an already-88.70%-accurate
    #   pretrained model. Result: accuracy crashed at the phase transition (88.70% -> 31.58%) and
    #   never recovered over 40 further epochs (final 22.70%, worse than every prior attempt).
    #   Classic catastrophic-forgetting-from-too-high-a-finetuning-LR, not a new discovery -
    #   standard transfer-learning/fine-tuning practice universally uses a much smaller LR than
    #   original training, never a full schedule restart at the same peak. Set to a value well
    #   below `learning_rate` (this project's own old, independently-verified-stable Adam LR of
    #   0.001 is a defensible starting point, not a guess) to test whether a genuinely gentle
    #   fine-tune - not a second full training run - is what the literature actually intends.
    epochs: int = 15
    num_batches: int = 128
    eval_interval: int = 3
    device: DeviceConfig = field(default_factory=DeviceConfig)
    crossbar: CrossbarConfig = field(default_factory=CrossbarConfig)
    fault: FaultConfig = field(default_factory=FaultConfig)
    # Optional second, independently-configured fault channel that fires on
    # the same wave schedule as `fault` (see __post_init__'s injection_interval
    # check below). None (default) preserves every existing config's behavior
    # exactly - a single fault type per run. Real gap this closes: dispatch.py's
    # apply_runtime_mitigation() only ever routes soft_mitigation through
    # monitor.drift_mask, but every pre-existing config injects exactly one
    # failure_model, so soft_mitigation and layer_reset (stuck-at-only via
    # remapping) never actually faced the fault type each is meant to defend
    # against in the same run as a genuine three-way comparison. Setting
    # secondary_fault to a FaultConfig with a different failure_model (e.g.
    # primary="density", secondary="drift") makes both fault types real and
    # simultaneous in one run instead of siloed across separate single-type runs.
    secondary_fault: FaultConfig | None = None
    mitigation: MitigationConfig = field(default_factory=MitigationConfig)
    retraining: RetrainingConfig = field(default_factory=RetrainingConfig)
    output_dir: str = "results"
    # False preserves prior behavior exactly (no checkpoint written). Opt-in per
    # config, not blanket - meant for runs whose result is intended to become a
    # reported/paper number (cheap for ResNet-9, ~7.5MB fp32), not every smoke test
    # or seed-sweep probe. checkpoint.pt is excluded from git (.gitignore) even
    # though the rest of results/ is tracked - weight files don't belong in git
    # history; a checkpoint is regenerable from the run's own recorded commit+seed.
    save_checkpoint: bool = False
    # "continuous": the existing loop - train through the faults the whole run.
    # "deployment" (the main study): load a pre-fault checkpoint, then run an
    #   INFERENCE-ONLY lifetime of len(fault.wave_densities) injection events
    #   (experiments/run.py::run_deployment); optimizers exist only inside a
    #   retraining policy's burst. Needs a pre-deploy checkpoint.
    regime: str = "continuous"
    # deployment regime: path to the pre_deploy.pt written by a save_pre_deploy_checkpoint
    # run (model weights + Fisher sidecar). Loaded into the base model BEFORE the
    # crossbar system is built, because references, health monitors and ADC calibration
    # are all captured at construction.
    pre_deploy_checkpoint: str | None = None
    # deployment regime: instead of a lifetime, train (float pretrain + analog epochs as
    # configured, FisherTracker accumulating) with NO fault injection, write
    # <output_dir>/pre_deploy.pt, and stop.
    save_pre_deploy_checkpoint: bool = False

    def __post_init__(self):
        if self.regime not in _REGIMES:
            raise ValueError(
                f"Unknown ExperimentConfig.regime: {self.regime!r} "
                "(expected 'continuous' or 'deployment')"
            )
        if self.simulator not in ("crosssim", "aihwkit"):
            raise ValueError(
                f"Unknown ExperimentConfig.simulator: {self.simulator!r} "
                "(expected 'crosssim' or 'aihwkit')"
            )
        if self.dataset not in ("cifar10", "cifar100", "mnist"):
            raise ValueError(
                f"Unknown ExperimentConfig.dataset: {self.dataset!r} "
                "(expected 'cifar10', 'cifar100' or 'mnist')"
            )
        expected_classes = 100 if self.dataset == "cifar100" else 10
        if self.num_classes != expected_classes:
            raise ValueError(
                f"ExperimentConfig.num_classes={self.num_classes!r} does not match dataset "
                f"{self.dataset!r} (expected {expected_classes})"
            )
        if self.model_arch not in ("resnet9", "tiny_cnn", "vgg6", "vit_tiny"):
            raise ValueError(
                f"Unknown ExperimentConfig.model_arch: {self.model_arch!r} "
                "(expected 'resnet9', 'tiny_cnn', 'vgg6' or 'vit_tiny')"
            )
        if self.train_samples < 1 or self.test_samples < 1:
            raise ValueError(
                f"ExperimentConfig.train_samples/test_samples must be >= 1, got "
                f"train_samples={self.train_samples!r}, test_samples={self.test_samples!r}"
            )
        if self.batch_size < 1:
            raise ValueError(f"ExperimentConfig.batch_size must be >= 1, got {self.batch_size!r}")
        if self.epochs < 1:
            raise ValueError(f"ExperimentConfig.epochs must be >= 1, got {self.epochs!r}")
        if self.num_batches < 1:
            raise ValueError(f"ExperimentConfig.num_batches must be >= 1, got {self.num_batches!r}")
        if self.eval_interval < 1:
            raise ValueError(
                f"ExperimentConfig.eval_interval must be >= 1, got {self.eval_interval!r}"
            )
        if self.learning_rate <= 0:
            raise ValueError(
                f"ExperimentConfig.learning_rate must be positive, got {self.learning_rate!r}"
            )
        if self.weight_decay < 0:
            raise ValueError(
                f"ExperimentConfig.weight_decay must be >= 0, got {self.weight_decay!r}"
            )
        if self.optimizer not in ("adam", "adamw", "sgd"):
            raise ValueError(
                f"Unknown ExperimentConfig.optimizer: {self.optimizer!r} (expected 'adam', 'adamw' or 'sgd')"
            )
        if self.lr_one_cycle and self.lr_cosine_schedule:
            raise ValueError(
                "ExperimentConfig.lr_one_cycle and lr_cosine_schedule are mutually exclusive "
                "(both anneal the LR over the run - pick one)"
            )
        if not (0.0 <= self.label_smoothing < 1.0):
            raise ValueError(
                f"ExperimentConfig.label_smoothing must be in [0, 1), got {self.label_smoothing!r}"
            )
        if self.grad_clip_norm < 0:
            raise ValueError(
                f"ExperimentConfig.grad_clip_norm must be >= 0, got {self.grad_clip_norm!r}"
            )
        if self.float_pretrain_epochs < 0:
            raise ValueError(
                "ExperimentConfig.float_pretrain_epochs must be >= 0, got "
                f"{self.float_pretrain_epochs!r}"
            )
        if self.float_pretrain_epochs >= self.epochs:
            raise ValueError(
                "ExperimentConfig.float_pretrain_epochs must be < epochs (there must be at "
                f"least one analog epoch left), got float_pretrain_epochs="
                f"{self.float_pretrain_epochs!r} epochs={self.epochs!r}"
            )
        if self.finetune_learning_rate < 0:
            raise ValueError(
                "ExperimentConfig.finetune_learning_rate must be >= 0, got "
                f"{self.finetune_learning_rate!r}"
            )
        if self.secondary_fault is not None:
            if self.secondary_fault.failure_model == self.fault.failure_model:
                raise ValueError(
                    "ExperimentConfig.secondary_fault.failure_model must differ from "
                    f"fault.failure_model (both are {self.fault.failure_model!r}) - "
                    "secondary_fault exists to combine two distinct fault types in one run, "
                    "not duplicate one"
                )
            if self.secondary_fault.injection_interval != self.fault.injection_interval:
                raise ValueError(
                    "ExperimentConfig.secondary_fault.injection_interval "
                    f"({self.secondary_fault.injection_interval!r}) must match "
                    f"fault.injection_interval ({self.fault.injection_interval!r}) - both "
                    "fault types fire on the same wave schedule so mitigation mechanisms face "
                    "them simultaneously, not offset"
                )
        self._validate_regime()

    def _validate_regime(self) -> None:
        r = self.retraining
        if self.save_pre_deploy_checkpoint and self.regime != "deployment":
            raise ValueError(
                "ExperimentConfig.save_pre_deploy_checkpoint requires regime='deployment', "
                f"got regime={self.regime!r}"
            )
        if self.save_pre_deploy_checkpoint and self.pre_deploy_checkpoint is not None:
            raise ValueError(
                "ExperimentConfig.save_pre_deploy_checkpoint and pre_deploy_checkpoint are "
                "mutually exclusive (one run either produces the checkpoint or consumes it), got "
                f"pre_deploy_checkpoint={self.pre_deploy_checkpoint!r}"
            )
        if self.regime == "continuous":
            if self.fault.wave_densities is not None:
                raise ValueError(
                    "ExperimentConfig.fault.wave_densities is only used by regime='deployment' "
                    "(continuous waves follow fault.injection_interval); remove it or set "
                    "regime='deployment'"
                )
            if self.pre_deploy_checkpoint is not None:
                raise ValueError(
                    "ExperimentConfig.pre_deploy_checkpoint is only used by "
                    f"regime='deployment', got regime={self.regime!r}"
                )
        else:
            if not self.save_pre_deploy_checkpoint:
                if self.pre_deploy_checkpoint is None:
                    raise ValueError(
                        "ExperimentConfig.regime='deployment' requires pre_deploy_checkpoint "
                        "(the pre-fault weights to deploy) unless save_pre_deploy_checkpoint=True"
                    )
                if self.fault.wave_densities is None:
                    raise ValueError(
                        "ExperimentConfig.regime='deployment' requires fault.wave_densities "
                        "(one density per wave event) unless save_pre_deploy_checkpoint=True"
                    )
                if self.fault.failure_model != "density":
                    raise ValueError(
                        "ExperimentConfig.regime='deployment' needs fault.failure_model="
                        f"'density' (wave_densities drives it), got {self.fault.failure_model!r}"
                    )
            if r.enable_fault_aware:
                raise ValueError(
                    "RetrainingConfig.enable_fault_aware is the legacy continuous-regime "
                    "baseline; regime='deployment' selects behavior via retraining.policy"
                )
        if self.regime == "continuous" and r.policy != "none":
            # The continuous-regime policy hook (experiments/run.py::_ContinuousPolicyRunner)
            # drives only capacity / co_mitigation, density-only waves, structural remap
            # + bursts. Anything else would silently not do what its name says.
            if r.policy not in ("capacity", "co_mitigation"):
                raise ValueError(
                    f"retraining.policy={r.policy!r} is not wired into the continuous regime "
                    "(only 'capacity' and 'co_mitigation' are); use regime='deployment'"
                )
            if self.fault.failure_model != "density" or self.secondary_fault is not None:
                raise ValueError(
                    "continuous-regime retraining policies need fault.failure_model='density' "
                    "and no secondary_fault (waves use a dedicated per-wave generator)"
                )
            m = self.mitigation
            if m.enable_soft_mitigation or m.enable_layer_reset or m.enable_predictive:
                raise ValueError(
                    "continuous-regime retraining policies replace health-gated dispatch: "
                    "mitigation.enable_soft_mitigation/enable_layer_reset/enable_predictive "
                    "must be False (they would silently never run)"
                )
            if self.simulator != "crosssim":
                raise ValueError(
                    "continuous-regime retraining policies need simulator='crosssim' "
                    "(stuck-weight freeze / living spares are CrossSim-only)"
                )
            if r.freeze_stuck_weights is False:
                raise ValueError(
                    "continuous-regime retraining policies need the stuck-weight freeze "
                    "(freeze_stuck_weights true/None), not False"
                )
        if r.enable_fault_aware and r.policy != "none":
            raise ValueError(
                "RetrainingConfig.enable_fault_aware (legacy) and retraining.policy="
                f"{r.policy!r} are mutually exclusive - pick one"
            )
        if (
            r.reference_mode in ("living", "living_orig")
            and self.crossbar.redundancy_scheme != "idealized_per_cell"
        ):
            raise ValueError(
                "RetrainingConfig.reference_mode='living' requires crossbar.redundancy_scheme="
                f"'idealized_per_cell' (a spare takes one cell's weight), got "
                f"{self.crossbar.redundancy_scheme!r}"
            )
        if r.policy == "co_mitigation":
            if not self.mitigation.enable_remapping:
                raise ValueError(
                    "retraining.policy='co_mitigation' allocates spares and needs "
                    "mitigation.enable_remapping=True"
                )
            if self.crossbar.redundancy_scheme != "idealized_per_cell":
                raise ValueError(
                    "retraining.policy='co_mitigation' needs crossbar.redundancy_scheme="
                    "'idealized_per_cell' (per-cell spares: ordering, budgets and living "
                    "references are per cell)"
                )
        if r.policy in ("capacity", "capacity_or_canary"):
            if not self.mitigation.enable_remapping:
                raise ValueError(
                    "retraining.policy='capacity' (burst only when a wave exceeds remap capacity) "
                    "needs mitigation.enable_remapping=True - without remapping every wave "
                    "exceeds capacity and the arm silently degenerates into 'periodic'"
                )
            if self.crossbar.redundancy_scheme != "idealized_per_cell":
                raise ValueError(
                    "retraining.policy='capacity' needs crossbar.redundancy_scheme="
                    "'idealized_per_cell': row/column repair leaves cells effectively stuck, so "
                    "'capacity exceeded' is not observable"
                )
        if r.policy in ("liu_oneshot", "bn_recal_only") and not self.mitigation.enable_remapping:
            raise ValueError(
                f"retraining.policy={r.policy!r} remaps every wave and needs "
                "mitigation.enable_remapping=True"
            )
        if r.policy == "liu_oneshot" and self.crossbar.redundancy_scheme != "idealized_per_cell":
            raise ValueError(
                "retraining.policy='liu_oneshot' needs crossbar.redundancy_scheme="
                "'idealized_per_cell' (significance-ordered remap is per cell)"
            )
        if r.remap_pacing == "paced" and r.expected_waves is None and self.regime == "continuous":
            raise ValueError(
                "RetrainingConfig.remap_pacing='paced' needs expected_waves in regime="
                "'continuous' (deployment derives it from fault.wave_densities)"
            )


def load_config(path: str | Path) -> ExperimentConfig:
    """Load an ExperimentConfig from a YAML file, converting nested dicts
    (device/crossbar/fault/mitigation) into their typed dataclasses so
    downstream code always has real fields to check, never raw dict keys."""
    raw = yaml.safe_load(Path(path).read_text())

    if "tile_shape" in raw.get("crossbar", {}) and raw["crossbar"]["tile_shape"] is not None:
        raw["crossbar"]["tile_shape"] = tuple(raw["crossbar"]["tile_shape"])

    return ExperimentConfig(
        **{
            k: v
            for k, v in raw.items()
            if k
            not in ("device", "crossbar", "fault", "secondary_fault", "mitigation", "retraining")
        },
        device=DeviceConfig(**raw.get("device", {})),
        crossbar=CrossbarConfig(**raw.get("crossbar", {})),
        fault=FaultConfig(**raw.get("fault", {})),
        secondary_fault=(
            FaultConfig(**raw["secondary_fault"])
            if raw.get("secondary_fault") is not None
            else None
        ),
        mitigation=MitigationConfig(**raw.get("mitigation", {})),
        retraining=RetrainingConfig(**raw.get("retraining", {})),
    )
