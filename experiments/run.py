"""CLI entrypoint for running one experiment arm.

Ports the training loop, results-dict schema, and plotting pattern from
Implementation.ipynb cells 5-6 - the only code in this repo proven to have
actually run against real data - rather than the never-executed main.py/
experiment.py. plot_creation.py's synthetic curve generator is not ported at
all: synthetic curves are not experimental results, and the original dissertation
notebook's reported numbers could not be reproduced.

Usage:
    uv run experiments/run.py --config experiments/configs/no_mitigation.yaml
    uv run experiments/run.py --config experiments/configs/no_mitigation.yaml \\
        --epochs 1 --num-batches 2 --train-samples 64 --test-samples 32
"""

from __future__ import annotations

import os

# Must precede the torch/sklearn imports below: both bundle their own
# libomp.dylib, and macOS aborts the process on the second dlopen unless this
# is set (aihwkit's import chain pulls in sklearn).
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import hashlib
import json
import logging
import time
from dataclasses import asdict, replace
from pathlib import Path

import torch
from torch import optim

from neurofault.config import ExperimentConfig, load_config
from neurofault.data import load_datasets
from neurofault.models.cnn import build_model
from neurofault.provenance import capture_environment, capture_git_info
from neurofault.retraining.canary import build_canary
from neurofault.retraining.policy_hook import WaveContext, build_policy
from neurofault.retraining.saliency import FisherTracker
from neurofault.system import FaultTolerantNeuromorphic

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def _build_optimizer(params, config: ExperimentConfig, lr: float | None = None):
    lr = config.learning_rate if lr is None else lr
    if config.optimizer == "sgd":
        return optim.SGD(
            params, lr=lr, momentum=config.momentum, nesterov=True, weight_decay=config.weight_decay
        )
    if config.optimizer == "adamw":
        return optim.AdamW(params, lr=lr, weight_decay=config.weight_decay)
    return optim.Adam(params, lr=lr, weight_decay=config.weight_decay)


def _build_scheduler(optimizer, config: ExperimentConfig, epochs: int, max_lr: float | None = None):
    # OneCycleLR steps once per BATCH, CosineAnnealingLR once per epoch - see
    # the call sites below. Validated mutually exclusive in config.py's
    # __post_init__, so only one of these branches is ever active.
    max_lr = config.learning_rate if max_lr is None else max_lr
    if config.lr_one_cycle:
        return optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=max_lr, steps_per_epoch=config.num_batches, epochs=epochs
        )
    if config.lr_cosine_schedule:
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    return None


_SNAPSHOT_INTERVAL_SECONDS = 2.0


def _atomic_write_json(obj, path: Path) -> None:
    """Temp file + rename, so a concurrent reader (rsync off a rented GPU box) never
    sees a half-written file - same discipline as _write_partial_results."""
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2))
    tmp.replace(path)


def _write_partial_results(metrics: dict, output_dir: Path) -> None:
    """Durable incremental snapshot, called every ~2 wall-clock seconds during
    training (not just once at the very end, the prior behavior). Real gap
    this closes (2026-09-29): a rented GPU instance destroyed mid-run - or
    even a plain crash - previously lost every batch of progress, since
    metrics.json/layer_health.json were only ever written after the full
    training loop returned. Atomic (temp file + rename) so a concurrent
    reader (e.g. a local rsync loop pulling this off a rented box) never sees a half-written file.
    """
    snapshot = {k: v for k, v in metrics.items() if k != "layer_health"}
    tmp = output_dir / "metrics.json.tmp"
    tmp.write_text(json.dumps(snapshot, indent=2))
    tmp.replace(output_dir / "metrics.json")

    layer_health_tmp = output_dir / "layer_health.json.tmp"
    layer_health_tmp.write_text(json.dumps(metrics["layer_health"], indent=2))
    layer_health_tmp.replace(output_dir / "layer_health.json")


def _pretrain_float(
    model, config: ExperimentConfig, trainloader, testloader, device, metrics, output_dir=None
) -> int:
    """Phase 1 of IBM's own published hardware-aware training methodology
    (Rasch et al. 2023, arXiv:2302.08469, Fig. 1) - train the plain,
    unpatched float model to convergence with NO crossbar simulation at all,
    before phase 2 (the caller's main loop) continues training through the
    analog forward pass. Mutates `model` in place; `system.py` then patches
    these already-trained weights instead of a random init. Logs into the
    same `metrics` dict as the main loop so plots/reports show one
    continuous curve - health/faults/mitigation fields are meaningless here
    (no crossbar exists yet), logged as inert sentinels (100.0/0), not
    omitted, so every row has the same schema.
    """
    optimizer = _build_optimizer(model.parameters(), config)
    scheduler = _build_scheduler(optimizer, config, epochs=config.float_pretrain_epochs)
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)

    last_snapshot = time.time()
    global_batch = 0
    for epoch in range(config.float_pretrain_epochs):
        epoch_start = time.time()
        model.train()
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            if batch_idx >= config.num_batches:
                break

            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            # Always measure the pre-clip gradient norm, even when clipping is off
            # (max_norm=inf is a standard no-op-but-still-measure trick) - previously
            # only computed, and only implicitly, when grad_clip_norm > 0, leaving no
            # record of it otherwise - a real gap during an earlier SGD-plateau
            # investigation.
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                config.grad_clip_norm if config.grad_clip_norm > 0 else float("inf"),
            )
            optimizer.step()
            if config.lr_one_cycle:
                scheduler.step()

            if global_batch % config.eval_interval == 0:
                accuracy = _evaluate_model(model, testloader, device)
                metrics["epoch"].append(epoch)
                metrics["batch"].append(global_batch)
                metrics["loss"].append(loss.item())
                metrics["accuracy"].append(accuracy)
                metrics["health"].append(100.0)
                metrics["faults"].append(0)
                metrics["soft_mitigations"].append(0)
                metrics["remappings"].append(0)
                metrics["layer_resets"].append(0)
                metrics["detection_cost"].append(0)
                metrics["grad_norm"].append(grad_norm.item())
                logger.info(
                    "[float pretrain] epoch=%d batch=%d loss=%.4f accuracy=%.2f%% grad_norm=%.4f",
                    epoch,
                    global_batch,
                    loss.item(),
                    accuracy,
                    grad_norm.item(),
                )

            if output_dir is not None and time.time() - last_snapshot >= _SNAPSHOT_INTERVAL_SECONDS:
                _write_partial_results(metrics, output_dir)
                last_snapshot = time.time()

            global_batch += 1

        if scheduler is not None and not config.lr_one_cycle:
            scheduler.step()
        metrics["epoch_seconds"].append(time.time() - epoch_start)

    if output_dir is not None:
        _write_partial_results(metrics, output_dir)

    return global_batch


def _collect_calibration_images(trainloader, num_images: int, device: torch.device) -> torch.Tensor:
    images = []
    n = 0
    for inputs, _ in trainloader:
        images.append(inputs)
        n += inputs.size(0)
        if n >= num_images:
            break
    return torch.cat(images, dim=0)[:num_images].to(device)


def _new_metrics() -> dict:
    return {
        "epoch": [],
        "batch": [],
        "loss": [],
        "accuracy": [],
        "health": [],
        "faults": [],
        "soft_mitigations": [],
        "remappings": [],
        "layer_resets": [],
        "detection_cost": [],  # modeled cell-touches (see health_monitor.py) - the
        #   honest number for comparing mitigation.detection_method="checksum" against
        #   "full_diff" (rows*cols per layer per check, always, for full_diff).
        "grad_norm": [],  # pre-clip L2 gradient norm, always measured (see
        #   _pretrain_float's comment) - same eval_interval cadence as loss/accuracy.
        "epoch_seconds": [],  # once per epoch (not eval_interval), continuous 0..epochs-1
        #   across pretrain+analog phases - real wall-clock, not an estimate read off a
        #   terminal by hand like every prior GPU timing figure in this project's history.
        "layer_health": [],  # popped by main() before metrics.json is written and saved
        #   separately as layer_health.json - the full per-crossbar detail
        #   system.get_health_report() already computes every eval interval and this
        #   loop previously discarded down to a single avg_health scalar.
    }


def _build_system(
    config: ExperimentConfig, base_model, trainloader, device: torch.device
) -> FaultTolerantNeuromorphic:
    """Calibrate (if configured) on base_model's CURRENT weights, then patch it into
    the crossbar system. Everything the system captures in __init__ (fault-free
    references, health-monitor baselines, ADC ranges) is therefore derived from
    whatever weights base_model holds right now - so callers load/train weights
    first, never after."""
    calibration_images = None
    if config.crossbar.calibrate_adc:
        # Calibrate AFTER any float pretraining, on base_model's real (trained,
        # not random-init) weights - the ADC's real internal signal range
        # depends on what the model actually computes, verified via this
        # local probes using the same warm-started-not-random-init
        # discipline.
        calibration_images = _collect_calibration_images(
            trainloader, config.crossbar.adc_calibration_images, device
        )
    return FaultTolerantNeuromorphic(base_model, config, calibration_images=calibration_images)


def _build_analog_optimizer(system, config: ExperimentConfig):
    """Optimizer + scheduler + criterion for the analog training phase."""
    analog_epochs = config.epochs - config.float_pretrain_epochs
    # Only override the LR when a pretrain phase actually ran - a config with
    # finetune_learning_rate set but float_pretrain_epochs=0 (unusual, but not
    # invalid) trains through the analog path from scratch as always, so it
    # should use `learning_rate` like every phase-0 config always has.
    finetune_lr = (
        config.finetune_learning_rate
        if config.float_pretrain_epochs > 0 and config.finetune_learning_rate > 0
        else None
    )
    optimizer = _build_optimizer(system.model.parameters(), config, lr=finetune_lr)
    scheduler = _build_scheduler(optimizer, config, epochs=analog_epochs, max_lr=finetune_lr)
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
    return optimizer, scheduler, criterion


def _train_analog(
    system,
    config: ExperimentConfig,
    trainloader,
    testloader,
    device: torch.device,
    metrics: dict,
    output_dir: Path | None,
    start_batch: int,
    optimizer,
    scheduler,
    criterion,
    *,
    inject_waves: bool,
    fisher: FisherTracker | None = None,
    policy_runner: _ContinuousPolicyRunner | None = None,
) -> int:
    """The analog (phase 2) training loop over config.float_pretrain_epochs..epochs.
    Returns the final global batch count.

    inject_waves=False runs it fault-free (pre-deploy checkpoint training); `fisher`
    (optional) is updated after every backward, before clipping. With
    inject_waves=True and fisher=None this is exactly the loop run_experiment has
    always run - same RNG consumption, same order of operations."""
    last_snapshot = time.time()
    global_batch = start_batch
    for epoch in range(config.float_pretrain_epochs, config.epochs):
        epoch_start = time.time()
        system.model.train()
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            if batch_idx >= config.num_batches:
                break

            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = system.forward(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            if fisher is not None:
                fisher.update()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                system.model.parameters(),
                config.grad_clip_norm if config.grad_clip_norm > 0 else float("inf"),
            )
            optimizer.step()
            if config.lr_one_cycle:
                scheduler.step()
            system.synchronize()  # pushes the update into the analog compute path, then
            #   re-pins stuck faults/mitigation corrections it would otherwise erase -
            #   without this, training silently never reaches backends that need it, and
            #   faults/mitigations silently self-heal within one batch (see
            #   system.py::synchronize()'s docstring - real bugs, not defensive code)

            if (
                inject_waves
                and not config.retraining.enable_fault_aware
                and global_batch % config.fault.injection_interval == 0
            ):
                if policy_runner is None:
                    system.inject_faults()
                else:
                    # retraining policy wave (capacity / co_mitigation): the runner injects
                    # with a dedicated generator, remaps, retrains in a burst on its own
                    # fresh optimizer; the main optimizer/scheduler are untouched.
                    policy_runner.on_wave(system, global_batch)
                    system.model.train()  # a burst/eval leaves the model in eval mode

            if global_batch % config.eval_interval == 0:
                accuracy = _evaluate(system, testloader, device)
                report = system.get_health_report()
                metrics["epoch"].append(epoch)
                metrics["batch"].append(global_batch)
                metrics["loss"].append(loss.item())
                metrics["accuracy"].append(accuracy)
                metrics["health"].append(report["avg_health"])
                metrics["faults"].append(report["total_faults_injected"])
                metrics["soft_mitigations"].append(report["soft_mitigations"])
                metrics["remappings"].append(report["remappings"])
                metrics["layer_resets"].append(report["layer_resets"])
                metrics["detection_cost"].append(
                    sum(s["detection_cost"] for s in report["layer_health"].values())
                )
                metrics["grad_norm"].append(grad_norm.item())
                metrics["layer_health"].append(
                    {"epoch": epoch, "batch": global_batch, "layers": report["layer_health"]}
                )
                logger.info(
                    "epoch=%d batch=%d loss=%.4f accuracy=%.2f%% health=%.2f%% grad_norm=%.4f",
                    epoch,
                    global_batch,
                    loss.item(),
                    accuracy,
                    report["avg_health"],
                    grad_norm.item(),
                )

            if output_dir is not None and time.time() - last_snapshot >= _SNAPSHOT_INTERVAL_SECONDS:
                _write_partial_results(metrics, output_dir)
                last_snapshot = time.time()

            global_batch += 1

        if scheduler is not None and not config.lr_one_cycle:
            scheduler.step()
        metrics["epoch_seconds"].append(time.time() - epoch_start)

    return global_batch


def _check_priors_file(config: ExperimentConfig) -> None:
    if (
        config.retraining.priors_path is not None
        and not Path(config.retraining.priors_path).is_file()
    ):
        # before any data/model work: D must never silently run uncalibrated
        raise FileNotFoundError(
            f"retraining.priors_path {config.retraining.priors_path!r} does not exist - run "
            "the GPU calibration sweep (experiments/calibration_sweep.py) and "
            "scripts/calibrate_retrain_controller.py first"
        )


class _ContinuousPolicyRunner:
    """Continuous-regime bridge for retraining policies (capacity / co_mitigation).

    Called by _train_analog at each wave step INSTEAD of system.inject_faults() (so with
    policy "none" this class is never constructed and the loop is unchanged). Per wave:
    dedicated per-wave generator (seed*1000+k, same as run_deployment, so every policy arm
    sees identical fault maps whatever its bursts consumed from the global RNG) -> inject ->
    apply_pins -> eval test + canary -> policy.on_wave (remap, BN recal, gate, burst) -> eval.

    Isolation from the main run: bursts (retraining/burst.py) build a FRESH SGD optimizer
    each time and call system.synchronize(); the main optimizer's state/momentum and the
    main LR scheduler are never touched or stepped by a burst. The caller restores
    model.train() afterwards. The main optimizer's momentum buffers are stale w.r.t. the
    burst's weight changes (a deliberate, documented simplification). Mitigation is
    policy-driven only: system.forward() skips health-gated dispatch for these runs.

    The gate reference is the canary right BEFORE this wave's injection (the network keeps
    training between waves, so the previous post-policy canary is stale).
    """

    def __init__(self, config, system, testloader, canary, fisher, device, output_dir):
        self.config, self.system, self.testloader = config, system, testloader
        self.canary, self.fisher, self.device = canary, fisher, device
        self.output_dir = output_dir
        self.policy = build_policy(config)
        self.events: list[dict] = []
        self.baseline_canary: float | None = None
        # Test accuracy after the LAST training batch (set by finish()). summary.final_test_acc
        # is the value right after the last wave (batch 15600 in the 3-wave config), not this.
        self.end_of_run_test_acc: float | None = None

    def _snapshot(self, done: bool) -> dict:
        ev = self.events
        after = [e["test_acc_after"] for e in ev]
        before = [e["test_acc_before"] for e in ev]
        return {
            "regime": "continuous",
            "policy": self.config.retraining.policy,
            "seed": self.config.seed,
            "baseline_canary_acc": self.baseline_canary,
            "complete": done,
            "summary": {
                "final_test_acc": after[-1] if after else None,
                "lifetime_mean_test_acc": sum(after) / len(after) if after else None,
                "worst_dip_test_acc": min(before) if before else None,
                "total_burst_batches": sum(e["burst_batches"] for e in ev),
                "total_remapped": sum(e["remapped"] for e in ev),
                "end_of_run_test_acc": self.end_of_run_test_acc,
            },
            "events": ev,
        }

    def _write(self, done: bool) -> None:
        if self.output_dir is not None:
            _atomic_write_json(self._snapshot(done), self.output_dir / "events.json")

    def on_wave(self, system, global_batch: int) -> None:
        cfg = self.config
        k = len(self.events)
        t0 = time.time()
        prev_stuck = {n: m.effective_stuck.clone() for n, m in system.monitors.items()}
        remaps_before = system.stats["remappings"]
        # Gate reference: canary BEFORE this wave's injection (evaluating after injection
        # would make the reference the damaged ~chance-level accuracy, so the gate never fires).
        canary_pre_injection = _evaluate_model(system.model, self.canary.canary_loader, self.device)
        _inject_wave(system, cfg.seed * 1000 + k)
        system.apply_pins()
        newly_stuck = sum(
            int((m.effective_stuck & ~prev_stuck[n]).sum()) for n, m in system.monitors.items()
        )
        map_hash = {n: _fault_map_hash(m) for n, m in system.monitors.items()}
        test_before = _evaluate_model(system.model, self.testloader, self.device)
        canary_before = _evaluate_model(system.model, self.canary.canary_loader, self.device)
        # No clean-deployed-network baseline exists in continuous training; the best honest
        # analogue is the canary just before the first wave.
        if self.baseline_canary is None:
            self.baseline_canary = canary_pre_injection
        ctx = WaveContext(
            system=system,
            config=cfg,
            wave_index=k,
            density=cfg.fault.density,
            device=self.device,
            testloader=self.testloader,
            canary=self.canary,
            fisher=self.fisher,
            prev_stuck=prev_stuck,
            test_acc_before=test_before,
            canary_acc_before=canary_before,
            prior_events=self.events,
            baseline_canary_acc=self.baseline_canary,
            gate_reference=canary_pre_injection,
        )
        policy_fields = self.policy.on_wave(ctx)
        system.model.eval()
        test_after = _evaluate_model(system.model, self.testloader, self.device)
        canary_after = _evaluate_model(system.model, self.canary.canary_loader, self.device)
        event = {
            "wave": k,
            "density": cfg.fault.density,
            "global_batch": global_batch,
            "fault_map_hash": map_hash,
            "newly_stuck": newly_stuck,
            "test_acc_before": test_before,
            "test_acc_after": test_after,
            "canary_acc_before": canary_before,
            "canary_acc_after": canary_after,
            "burst_batches": 0,
            "optimizer_steps": 0,
            "remapped": system.stats["remappings"] - remaps_before,
            **policy_fields,
            "wall_seconds": time.time() - t0,
        }
        self.events.append(event)
        logger.info(
            "[continuous-policy] wave %d batch %d newly_stuck=%d test %.2f%% -> %.2f%% "
            "canary %.2f%% -> %.2f%% burst_batches=%d",
            k,
            global_batch,
            newly_stuck,
            test_before,
            test_after,
            canary_before,
            canary_after,
            event["burst_batches"],
        )
        self._write(done=False)

    def finish(self, end_of_run_test_acc: float | None = None) -> None:
        self.end_of_run_test_acc = end_of_run_test_acc
        self._write(done=True)


def _train_pipeline(
    config: ExperimentConfig,
    output_dir: Path | None,
    *,
    inject_waves: bool,
    track_fisher: bool,
) -> tuple[dict, FaultTolerantNeuromorphic, FisherTracker | None]:
    """Float pretrain -> build system -> analog training. Shared by run_experiment
    (continuous regime) and run_predeploy (fault-free, Fisher-tracked)."""
    torch.manual_seed(config.seed)

    trainloader, testloader = load_datasets(config)
    policy_active = inject_waves and config.retraining.policy != "none"
    canary = None
    if policy_active:
        _check_priors_file(config)
        # Only for retraining policies (policy "none" builds nothing here and consumes no
        # RNG). The canary is held out of ALL training, main stream included, so canary
        # accuracy stays a held-out signal; the main stream is therefore the
        # (train_samples - canary_size)-image remainder, shuffled by the global RNG as
        # load_datasets' loader is. Bursts draw from canary.burst_loader (same images,
        # private shuffle generator) - they do not consume the main stream's data order.
        canary = build_canary(config)
        trainloader = torch.utils.data.DataLoader(
            torch.utils.data.Subset(trainloader.dataset.dataset, canary.burst_indices),
            batch_size=config.batch_size,
            shuffle=True,
        )
    # CrossSim's AnalogLayer._set_device() raises if useGPU=True but the
    # underlying torch layer's own device isn't 'cuda' at conversion time
    # (verified hands-on on a real rented GPU box - this path was never testable before) - the model
    # must
    # move to the target device *before* patch_layers() wraps its layers,
    # not after.
    device = torch.device("cuda" if config.crossbar.use_gpu else "cpu")
    base_model = build_model(config).to(device)

    metrics = _new_metrics()

    pretrain_final_batch = 0
    if config.float_pretrain_epochs > 0:
        pretrain_final_batch = _pretrain_float(
            base_model, config, trainloader, testloader, device, metrics, output_dir
        )

    if config.float_pretrain_epochs > 0 and device.type == "cuda":
        # Real OOM hit here: CUDA's caching allocator fragments
        # over the pretrain phase's many batches (worse the more epochs
        # pretrained), leaving no large-enough contiguous block for the
        # calibration pass's temporary model copies / the patched model's
        # large crossbars right after - even though total free memory would
        # technically be enough. Clearing the cache at this phase transition
        # is standard PyTorch practice for exactly this handoff pattern.
        torch.cuda.empty_cache()

    system = _build_system(config, base_model, trainloader, device)
    optimizer, scheduler, criterion = _build_analog_optimizer(system, config)

    if inject_waves and config.retraining.enable_fault_aware:
        # Bake a fixed, known defect map in once, before any training - the
        # dominant literature baseline: gradient descent then learns weights that compensate for
        # these
        # specific, unchanging faults, instead of reacting to faults at
        # runtime. No periodic re-injection below for this mode. Pinning the
        # stuck cells so they survive synchronize() is now automatic (see
        # system.py's inject_faults()/synchronize()) - no separate capture call.
        system.inject_faults()

    fisher = (
        FisherTracker.from_handles(system.handles, config.retraining.fisher_ema_decay)
        if track_fisher or (policy_active and config.retraining.policy == "co_mitigation")
        else None
    )
    policy_runner = None
    if policy_active:
        system.fisher = fisher  # remap_order='saliency' reads it (None for capacity/C)
        policy_runner = _ContinuousPolicyRunner(
            config, system, testloader, canary, fisher, device, output_dir
        )

    global_batch = _train_analog(
        system,
        config,
        trainloader,
        testloader,
        device,
        metrics,
        output_dir,
        pretrain_final_batch,
        optimizer,
        scheduler,
        criterion,
        inject_waves=inject_waves,
        fisher=fisher,
        policy_runner=policy_runner,
    )
    logger.info("Analog training finished at global batch %d", global_batch)
    if policy_runner is not None:
        # End-of-training test accuracy (after the last batch, not after the last wave).
        policy_runner.finish(end_of_run_test_acc=_evaluate(system, testloader, device))

    if output_dir is not None:
        _write_partial_results(metrics, output_dir)

    return metrics, system, fisher


def run_experiment(config: ExperimentConfig, output_dir: Path | None = None) -> dict:
    """`output_dir` is only used for an optional final-epoch checkpoint
    (config.save_checkpoint) - metrics/config/logs are still written by the
    caller (main()). None (default) means "don't checkpoint even if
    save_checkpoint is set" - existing direct callers (tests/invariants/)
    never pass it and never want checkpoint files appearing under them."""
    metrics, system, _ = _train_pipeline(config, output_dir, inject_waves=True, track_fisher=False)

    if config.save_checkpoint and output_dir is not None:
        torch.save(system.model.state_dict(), output_dir / "checkpoint.pt")
        logger.info("Checkpoint written to %s", output_dir / "checkpoint.pt")

    return metrics


def _config_hash(config: ExperimentConfig) -> str:
    return hashlib.sha256(json.dumps(asdict(config), sort_keys=True).encode()).hexdigest()


PRE_DEPLOY_FILENAME = "pre_deploy.pt"


def run_predeploy(config: ExperimentConfig, output_dir: Path | None = None) -> dict:
    """Deployment regime, checkpoint-producing mode (config.save_pre_deploy_checkpoint):
    float pretrain + analog training exactly as configured, but with NO fault injection
    and a FisherTracker accumulating gradient^2 through the analog phase, then write
    `<output_dir>/pre_deploy.pt` and stop. The deployment lifetime (run_deployment) starts
    from this clean pre-fault model. Contents: model_state_dict (the patched model's,
    loadable into a fresh build_model(config)), fisher_state (FisherTracker.state_dict()), the
    producing config's hash, and the final test accuracy."""
    metrics, system, fisher = _train_pipeline(
        config, output_dir, inject_waves=False, track_fisher=True
    )
    _, testloader = load_datasets(config)
    device = torch.device("cuda" if config.crossbar.use_gpu else "cpu")
    final_acc = _evaluate(system, testloader, device)
    if output_dir is not None:
        path = output_dir / PRE_DEPLOY_FILENAME
        torch.save(
            {
                "model_state_dict": {
                    k: v.detach().cpu() for k, v in system.model.state_dict().items()
                },
                "fisher_state": fisher.state_dict(),
                "config_hash": _config_hash(config),
                "final_test_acc": final_acc,
            },
            path,
        )
        logger.info("Pre-deploy checkpoint written to %s (test acc %.2f%%)", path, final_acc)
    return metrics


# Nominal spacing between wave events, in batch-equivalents. 5200 is the injection_interval
# the existing continuous-regime runs use (13 epochs x 400 batches ~ one wave per ~13
# epochs), so deployment lifetimes are expressed in the same unit as the continuous
# results. It is NOMINAL: a deployment has no batches, only inference between waves, so
# this number only weights lifetime-mean accuracy; it is not measured time.
NOMINAL_INTER_WAVE_BATCHES = 5200


def _fault_map_hash(monitor) -> str:
    return hashlib.sha256(monitor.stuck_mask.cpu().numpy().tobytes()).hexdigest()[:16]


def _inject_wave(system, wave_seed: int) -> int:
    """Inject one wave with a dedicated, per-wave RNG so every policy arm sees identical
    fault maps regardless of how much global RNG its bursts consumed."""
    # The generator must live on the crossbar tensors' device (torch requirement, see
    # faults/injection.py). Every arm of a run uses the same device, so fault maps stay
    # identical across arms; a CUDA map differs from a CPU map for the same seed.
    device = next(iter(system.monitors.values())).stuck_mask.device
    g = torch.Generator(device=device)
    g.manual_seed(wave_seed)
    return system.inject_faults(generator=g)


def _load_deployment_system(config: ExperimentConfig, device: torch.device):
    """Pre-deploy checkpoint -> crossbar system (+ Fisher tracker attached as
    system.fisher, or None). Shared by run_deployment and calibration_sweep so the sweep
    rebuilds EXACTLY the system a deployment starts from. The checkpoint is loaded into
    base_model BEFORE the system exists: references, health monitors and ADC calibration
    are all captured in FaultTolerantNeuromorphic.__init__."""
    ckpt = torch.load(config.pre_deploy_checkpoint, map_location="cpu", weights_only=False)
    trainloader, testloader = load_datasets(config)
    base_model = build_model(config)
    base_model.load_state_dict(ckpt["model_state_dict"], strict=True)
    base_model.to(device)

    system = _build_system(config, base_model, trainloader, device)
    fisher = None
    if ckpt.get("fisher_state") is not None:
        fisher = FisherTracker.from_handles(system.handles, config.retraining.fisher_ema_decay)
        fisher.load_state_dict(ckpt["fisher_state"])
    system.fisher = fisher  # remap_order='saliency' reads it (None -> a clear error there)
    return system, ckpt, fisher, trainloader, testloader


def run_deployment(config: ExperimentConfig, output_dir: Path | None = None) -> dict:
    """Deployment-regime lifetime: load the pre-fault checkpoint, then K wave events
    (config.fault.wave_densities), inference only.

    Order matters: the checkpoint is loaded into base_model BEFORE the crossbar system
    exists, because references, health monitors and ADC calibration are all captured in
    FaultTolerantNeuromorphic.__init__. Each wave: dedicated generator (seed*1000+k) ->
    inject at that wave's density -> record fault-map hash -> eval test+canary -> policy
    hook (the only place an optimizer may exist) -> eval again. No optimizer is
    constructed here; mitigation never runs inside eval (it lives in system.forward()).

    Writes events.json (atomically, after every wave) when output_dir is given and
    returns a metrics dict shaped like run_experiment's (one row per wave, post-policy).
    """
    torch.manual_seed(config.seed)
    device = torch.device("cuda" if config.crossbar.use_gpu else "cpu")

    _check_priors_file(config)
    system, ckpt, fisher, _, testloader = _load_deployment_system(config, device)
    canary = build_canary(config)
    policy = build_policy(config)
    system.model.eval()  # inference-only for the whole lifetime; a policy burst manages its own mode

    base_fault = config.fault
    densities = list(base_fault.wave_densities)
    events: list[dict] = []
    metrics = _new_metrics()

    baseline_test = _evaluate_model(system.model, testloader, device)
    baseline_canary = _evaluate_model(system.model, canary.canary_loader, device)
    logger.info(
        "[deploy] baseline test=%.2f%% canary=%.2f%% (checkpoint acc %.2f%%)",
        baseline_test,
        baseline_canary,
        ckpt.get("final_test_acc", float("nan")),
    )

    def _snapshot_events(done: bool) -> dict:
        after = [e["test_acc_after"] for e in events]
        before = [e["test_acc_before"] for e in events]
        return {
            "regime": "deployment",
            "policy": config.retraining.policy,
            "seed": config.seed,
            "nominal_inter_wave_batches": NOMINAL_INTER_WAVE_BATCHES,
            "checkpoint_config_hash": ckpt.get("config_hash"),
            "checkpoint_final_test_acc": ckpt.get("final_test_acc"),
            "baseline_test_acc": baseline_test,
            "baseline_canary_acc": baseline_canary,
            "complete": done,
            # Lifetime-mean: each wave's post-policy accuracy holds for one nominal
            # inter-wave interval (equal intervals, so a plain mean). Worst dip is the
            # lowest PRE-policy accuracy, i.e. the damage a wave did before any response.
            "summary": {
                "final_test_acc": after[-1] if after else None,
                "lifetime_mean_test_acc": sum(after) / len(after) if after else None,
                "worst_dip_test_acc": min(before) if before else None,
                "total_burst_batches": sum(e["burst_batches"] for e in events),
                "total_remapped": sum(e["remapped"] for e in events),
            },
            "events": events,
        }

    for k, density in enumerate(densities):
        wave_start = time.time()
        # system holds its config by reference and inject_faults() reads config.fault at
        # call time, so a per-wave density is a dataclasses.replace of the fault config -
        # no change to system.py needed.
        system.config = replace(system.config, fault=replace(base_fault, density=density))
        # EFFECTIVE stuck (stuck & ~remapped), not raw stuck_mask: a spare that failed
        # again is a new fault for the policy (and for newly_stuck below); the raw mask
        # would miss it because that position was already stuck_mask=True. The fault-map
        # HASH below stays on the raw stuck_mask - it is the identity of the INJECTED
        # maps, which must not depend on what a policy remapped.
        prev_stuck = {name: m.effective_stuck.clone() for name, m in system.monitors.items()}
        remaps_before = system.stats["remappings"]

        _inject_wave(system, config.seed * 1000 + k)
        system.apply_pins()  # make the wave visible in the analog cores (no synchronize happened)

        newly_stuck = sum(
            int((m.effective_stuck & ~prev_stuck[name]).sum())
            for name, m in system.monitors.items()
        )
        map_hash = {name: _fault_map_hash(m) for name, m in system.monitors.items()}
        test_before = _evaluate_model(system.model, testloader, device)
        canary_before = _evaluate_model(system.model, canary.canary_loader, device)

        ctx = WaveContext(
            system=system,
            config=config,
            wave_index=k,
            density=density,
            device=device,
            testloader=testloader,
            canary=canary,
            fisher=fisher,
            prev_stuck=prev_stuck,
            test_acc_before=test_before,
            canary_acc_before=canary_before,
            prior_events=events,
            baseline_canary_acc=baseline_canary,
        )
        policy_fields = policy.on_wave(ctx)
        system.model.eval()  # a policy must hand back an eval-mode model

        test_after = _evaluate_model(system.model, testloader, device)
        canary_after = _evaluate_model(system.model, canary.canary_loader, device)
        event = {
            "wave": k,
            "density": density,
            "nominal_batch_equivalent": k * NOMINAL_INTER_WAVE_BATCHES,
            "fault_map_hash": map_hash,
            "newly_stuck": newly_stuck,
            "test_acc_before": test_before,
            "test_acc_after": test_after,
            "canary_acc_before": canary_before,
            "canary_acc_after": canary_after,
            "burst_batches": 0,  # overridden by the policy's value below
            "optimizer_steps": 0,
            "remapped": system.stats["remappings"] - remaps_before,
            **policy_fields,
        }
        event["wall_seconds"] = time.time() - wave_start
        events.append(event)
        logger.info(
            "[deploy] wave %d density=%.3f newly_stuck=%d test %.2f%% -> %.2f%% canary %.2f%% -> %.2f%%",
            k,
            density,
            newly_stuck,
            test_before,
            test_after,
            canary_before,
            canary_after,
        )

        report = system.get_health_report()
        metrics["epoch"].append(k)
        metrics["batch"].append(k * NOMINAL_INTER_WAVE_BATCHES)
        metrics["loss"].append(0.0)  # inert sentinel (no training loss in a lifetime), like
        metrics["accuracy"].append(test_after)  # _pretrain_float's inert health/faults sentinels
        metrics["health"].append(report["avg_health"])
        metrics["faults"].append(report["total_faults_injected"])
        metrics["soft_mitigations"].append(report["soft_mitigations"])
        metrics["remappings"].append(report["remappings"])
        metrics["layer_resets"].append(report["layer_resets"])
        metrics["detection_cost"].append(
            sum(s["detection_cost"] for s in report["layer_health"].values())
        )
        metrics["grad_norm"].append(0.0)
        metrics["epoch_seconds"].append(event["wall_seconds"])
        metrics["layer_health"].append(
            {"epoch": k, "batch": k * NOMINAL_INTER_WAVE_BATCHES, "layers": report["layer_health"]}
        )

        if output_dir is not None:
            _atomic_write_json(_snapshot_events(done=False), output_dir / "events.json")

    final = _snapshot_events(done=True)
    if output_dir is not None:
        _atomic_write_json(final, output_dir / "events.json")
        _write_partial_results(metrics, output_dir)
    metrics["events"] = final
    return metrics


def _evaluate_model(model, testloader, device: torch.device) -> float:
    was_training = model.training
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for inputs, targets in testloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            predicted = outputs.argmax(dim=1)
            correct += (predicted == targets).sum().item()
            total += targets.size(0)
    if was_training:
        model.train()
    return 100.0 * correct / total if total > 0 else 0.0


def _evaluate(system: FaultTolerantNeuromorphic, testloader, device: torch.device) -> float:
    return _evaluate_model(system.model, testloader, device)


def plot_single_run(metrics: dict, output_path: Path, title: str) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(20, 10))

    axes[0, 0].plot(metrics["batch"], metrics["accuracy"])
    axes[0, 0].set(title="Accuracy", xlabel="Batch", ylabel="Accuracy (%)")

    axes[0, 1].plot(metrics["batch"], metrics["health"])
    axes[0, 1].set(title="Health", xlabel="Batch", ylabel="Health (%)")

    axes[0, 2].plot(metrics["batch"], metrics["grad_norm"])
    axes[0, 2].set(title="Gradient norm (pre-clip)", xlabel="Batch", ylabel="L2 norm")

    axes[1, 0].plot(metrics["batch"], metrics["loss"])
    axes[1, 0].set(title="Loss", xlabel="Batch", ylabel="Loss")

    axes[1, 1].plot(metrics["batch"], metrics["faults"], label="Faults")
    axes[1, 1].plot(metrics["batch"], metrics["soft_mitigations"], label="Soft mitigations")
    axes[1, 1].plot(metrics["batch"], metrics["remappings"], label="Remappings")
    axes[1, 1].plot(metrics["batch"], metrics["layer_resets"], label="Layer resets")
    axes[1, 1].set(title="Faults and mitigations", xlabel="Batch", ylabel="Count")
    axes[1, 1].legend()

    axes[1, 2].plot(range(len(metrics["epoch_seconds"])), metrics["epoch_seconds"])
    axes[1, 2].set(title="Wall-clock time per epoch", xlabel="Epoch", ylabel="Seconds")

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--num-batches", type=int)
    parser.add_argument("--train-samples", type=int)
    parser.add_argument("--test-samples", type=int)
    parser.add_argument("--simulator", choices=["crosssim", "aihwkit"])
    parser.add_argument("--dataset", choices=["cifar10", "cifar100", "mnist"])
    parser.add_argument("--seed", type=int)
    parser.add_argument(
        "--pre-deploy-checkpoint",
        help="deployment regime: path to a pre_deploy.pt (overrides the config's placeholder)",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    overrides = {
        k: v
        for k, v in {
            "epochs": args.epochs,
            "num_batches": args.num_batches,
            "train_samples": args.train_samples,
            "test_samples": args.test_samples,
            "simulator": args.simulator,
            "dataset": args.dataset,
            # keep the head width consistent when --dataset switches CIFAR-10 <-> CIFAR-100
            "num_classes": (
                None if args.dataset is None else 100 if args.dataset == "cifar100" else 10
            ),
            "seed": args.seed,
            "pre_deploy_checkpoint": args.pre_deploy_checkpoint,
        }.items()
        if v is not None
    }
    if args.seed is not None:
        # Distinct results/<name>/ per seed - otherwise a seed sweep would
        # have every run overwrite the same directory.
        overrides["name"] = f"{config.name}_seed{args.seed}"
    if overrides:
        config = replace(config, **overrides)

    output_dir = Path(config.output_dir) / config.name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Every run keeps its full raw log automatically from here on - previously only
    # true for the 2 of 67 past runs where stdout happened to be manually piped
    # through `nohup ... > log` on a rented GPU box.
    # Attached/detached per call, not left on the root logger forever - a
    # process that calls main() more than once (a future sweep script, this
    # project's own tests) would otherwise keep writing every later run's
    # output into this run's run.log too.
    file_handler = logging.FileHandler(output_dir / "run.log")
    file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    logging.getLogger().addHandler(file_handler)

    try:
        git_info = capture_git_info()
        if git_info.get("dirty"):
            logger.warning(
                "Working tree has uncommitted changes - this run is not exactly "
                "reproducible from git history alone (see run_metadata.json's git.dirty)."
            )

        # Written BEFORE training starts, not just at the end (real gap found
        # 2026-09-29: a rented GPU instance destroyed mid-run took every result
        # with it, since nothing was durable until the full run returned). Config
        # and git/environment provenance are already fully known at this point -
        # no reason to wait. Rewritten once more at the end below, adding
        # `timing` (which genuinely isn't known until the run finishes).
        (output_dir / "config.json").write_text(json.dumps(asdict(config), indent=2))
        (output_dir / "run_metadata.json").write_text(
            json.dumps({"git": git_info, "environment": capture_environment()}, indent=2)
        )

        start_time = time.time()
        if config.regime == "deployment" and config.save_pre_deploy_checkpoint:
            metrics = run_predeploy(config, output_dir)
        elif config.regime == "deployment":
            metrics = run_deployment(config, output_dir)
            metrics.pop("events")  # already written to events.json by run_deployment
        else:
            metrics = run_experiment(config, output_dir)
        total_seconds = time.time() - start_time

        layer_health_snapshots = metrics.pop("layer_health")
        (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
        (output_dir / "layer_health.json").write_text(json.dumps(layer_health_snapshots, indent=2))
        (output_dir / "config.json").write_text(json.dumps(asdict(config), indent=2))
        (output_dir / "run_metadata.json").write_text(
            json.dumps(
                {
                    "git": git_info,
                    "environment": capture_environment(),
                    "timing": {"total_seconds": total_seconds},
                },
                indent=2,
            )
        )
        plot_single_run(metrics, output_dir / "plots.png", title=config.name)

        logger.info("Results written to %s", output_dir)
    finally:
        logging.getLogger().removeHandler(file_handler)
        file_handler.close()

    # Post-experiment sanity gate -
    # the natural moment results "come back" is exactly when a run finishes,
    # so this needs no extra manual step. Imported from neurofault.validation
    # (an installed package), not experiments.validate_results - verified
    # directly that `experiments` itself isn't reliably importable under
    # this project's own bare-script invocation style (`uv run
    # experiments/run.py ...`), only under its registered console script.
    from neurofault.validation.report import validate_results_dir

    validate_results_dir(output_dir)


if __name__ == "__main__":
    main()
