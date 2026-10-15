"""Retraining burst: a short, fresh-optimizer training episode after a fault wave.

The only place an optimizer may exist in the deployment regime. Design points, each
load-bearing:

* Bursts call `system.model` directly, never `system.forward` - forward() runs
  check_health() every N calls, whose dispatch would remap/reset behind the policy's
  back and let default mitigation override a controller's allocation. Mitigation in
  deployment is policy-driven only (system.remap_now).
* `system.synchronize()` after EVERY step: it pushes the update into the analog cores
  and then re-asserts stuck pins and restores the frozen .weight of effectively-stuck
  cells (system.apply_pins), so stuck cells neither heal nor accumulate drifting
  weights during the burst.
* Fresh SGD per burst (momentum/nesterov/weight_decay from the config, same recipe as
  experiments/run.py::_build_optimizer - duplicated here because src/ cannot import
  experiments/), cosine LR from `lr` toward 0 over the PLANNED horizon (n_batches, or
  the stopper's t_max, so an early-stopped burst has simply not decayed fully).
* Optimizer steps are counted at the actual optimizer.step() call, independently of
  the batch counter, so "burst_batches == optimizer_steps" in events.json is a real
  check rather than a tautology.
* The train loader must be the canary-excluded burst loader (retraining/canary.py);
  the optional `canary` loader is evaluated through the clean transform.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
from torch import optim

from neurofault.config import ExperimentConfig
from neurofault.retraining.canary import canary_sigma
from neurofault.retraining.recovery_model import PatienceStopper
from neurofault.retraining.saliency import FisherTracker


@dataclass
class BurstResult:
    batches: int  # training batches consumed
    optimizer_steps: int  # optimizer.step() calls actually made (== batches)
    canary_trace: list[list[float]] = field(default_factory=list)  # [[batch, canary_acc], ...]
    stop_reason: str = "fixed"  # "fixed" | "plateau" | "t_max" | "plateau+anneal"
    restored_best: bool = False  # restore_best: the best earlier weights were loaded back
    best_canary: float | None = None  # restore_best: best canary reading seen in the burst


def evaluate_accuracy(model: torch.nn.Module, loader, device) -> float:
    """Top-1 accuracy (%) with the model in eval mode; restores the prior mode."""
    was_training = model.training
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for inputs, targets in loader:
            outputs = model(inputs.to(device))
            correct += (outputs.argmax(dim=1).cpu() == targets).sum().item()
            total += targets.size(0)
    if was_training:
        model.train()
    return 100.0 * correct / total if total else 0.0


def _clone_state(model: torch.nn.Module) -> dict:
    """Detached CPU copy of the model state (for restore_best)."""
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def _build_burst_optimizer(params, config: ExperimentConfig, lr: float):
    if config.optimizer == "sgd":
        return optim.SGD(
            params, lr=lr, momentum=config.momentum, nesterov=True, weight_decay=config.weight_decay
        )
    if config.optimizer == "adamw":
        return optim.AdamW(params, lr=lr, weight_decay=config.weight_decay)
    return optim.Adam(params, lr=lr, weight_decay=config.weight_decay)


def run_burst(
    system,
    train_loader,
    *,
    lr: float,
    n_batches: int | None = None,
    stopper: PatienceStopper | None = None,
    canary=None,
    eval_every: int | None = None,
    fisher: FisherTracker | None = None,
    config: ExperimentConfig,
    device,
) -> BurstResult:
    """Train `system.model` for `n_batches` (fixed) or until `stopper` says stop.

    Exactly one of n_batches / stopper. With a stopper, `canary` (a DataLoader) is
    required and evaluated every stopper.eval_every batches, feeding the stopper; with
    n_batches, a `canary` + `eval_every` just records a trace. canary_trace entries are
    [batch, accuracy] with batch counted from 1 (after that many steps)."""
    if (n_batches is None) == (stopper is None):
        raise ValueError("run_burst needs exactly one of n_batches or stopper")
    if stopper is not None:
        if canary is None:
            raise ValueError("run_burst with a stopper needs a canary loader")
        horizon, eval_every = stopper.t_max, stopper.eval_every
    else:
        if n_batches < 0:
            raise ValueError(f"n_batches must be >= 0, got {n_batches!r}")
        horizon = n_batches
    if (canary is None) != (eval_every is None) and stopper is None:
        raise ValueError("canary and eval_every must be given together")
    if horizon == 0:
        return BurstResult(0, 0, [], "fixed")

    model = system.model
    optimizer = _build_burst_optimizer(model.parameters(), config, lr)
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)

    result = BurstResult(0, 0, [], "fixed")
    rt = config.retraining
    anneal_on_stop = rt.anneal_on_stop and stopper is not None
    restore_best = rt.restore_best and canary is not None and eval_every is not None
    best_acc: float | None = None
    best_state: dict | None = None
    model.train()
    try:
        done = False
        while not done:
            progressed = False
            for inputs, targets in train_loader:
                progressed = True
                for group in optimizer.param_groups:
                    group["lr"] = lr * 0.5 * (1.0 + math.cos(math.pi * result.batches / horizon))
                inputs, targets = inputs.to(device), targets.to(device)
                optimizer.zero_grad()
                loss = criterion(model(inputs), targets)
                loss.backward()
                if fisher is not None:
                    fisher.update()  # raw gradient: after backward, before clipping
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    config.grad_clip_norm if config.grad_clip_norm > 0 else float("inf"),
                )
                optimizer.step()
                result.optimizer_steps += 1
                system.synchronize()
                result.batches += 1

                if (
                    eval_every is not None
                    and canary is not None
                    and (result.batches % eval_every == 0)
                ):
                    acc = evaluate_accuracy(model, canary, device)
                    result.canary_trace.append([result.batches, acc])
                    if restore_best and (best_acc is None or acc > best_acc):
                        best_acc, best_state = acc, _clone_state(model)
                    if stopper is not None and stopper.update(acc):
                        result.stop_reason = stopper.reason
                        done = True
                        break
                if result.batches >= horizon:
                    if stopper is not None:
                        result.stop_reason = stopper.reason or "t_max"
                    done = True
                    break
            if not progressed:
                raise ValueError("run_burst: train_loader yielded no batches")

        # D2 switch: after a plateau stop, anneal the LR to 0 over `anneal_steps` more steps.
        if anneal_on_stop and result.stop_reason == "plateau":
            # Start from the LR the cosine schedule would use at the next step.
            start_lr = lr * 0.5 * (1.0 + math.cos(math.pi * result.batches / horizon))
            n_anneal = rt.anneal_steps
            step_in_anneal = 0
            while step_in_anneal < n_anneal:
                progressed = False
                for inputs, targets in train_loader:
                    progressed = True
                    for group in optimizer.param_groups:
                        group["lr"] = (
                            start_lr * 0.5 * (1.0 + math.cos(math.pi * step_in_anneal / n_anneal))
                        )
                    inputs, targets = inputs.to(device), targets.to(device)
                    optimizer.zero_grad()
                    loss = criterion(model(inputs), targets)
                    loss.backward()
                    if fisher is not None:
                        fisher.update()
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        config.grad_clip_norm if config.grad_clip_norm > 0 else float("inf"),
                    )
                    optimizer.step()
                    result.optimizer_steps += 1
                    system.synchronize()
                    result.batches += 1
                    step_in_anneal += 1
                    if step_in_anneal >= n_anneal:
                        break
                if not progressed:
                    raise ValueError("run_burst: train_loader yielded no batches")
            for group in optimizer.param_groups:
                group["lr"] = 0.0  # the anneal ends at LR 0
            result.stop_reason = "plateau+anneal"

        # D2 switch: if the final reading is below the best one, put the best weights back.
        if restore_best:
            if result.canary_trace and result.canary_trace[-1][0] == result.batches:
                final_acc = result.canary_trace[-1][1]
            else:
                final_acc = evaluate_accuracy(model, canary, device)
                result.canary_trace.append([result.batches, final_acc])
                if best_acc is None or final_acc > best_acc:
                    best_acc, best_state = final_acc, _clone_state(model)
            result.best_canary = best_acc
            # Restore only if the best reading beats the final one by more than one canary standard
            # error (D2 amendment): a smaller gap is noise, and the annealed final state is kept.
            if best_acc is not None and (best_acc - final_acc) > canary_sigma(
                best_acc, rt.canary_size
            ):
                model.load_state_dict(best_state)
                system.synchronize()  # re-apply stuck pins after the restore
                result.restored_best = True
    finally:
        model.eval()
    return result
