"""The seam between run_deployment()'s lifetime loop and retraining policies.

run_deployment() owns everything that must be IDENTICAL across arms (fault maps,
evals, event bookkeeping); a policy owns everything that differs (structural
mitigation, BN recalibration, bursts). The loop calls `policy.on_wave(ctx)` once per
wave, after injection and the pre-policy evals, and merges the returned dict into the
wave's event record. Optimizers may only be constructed inside a policy.

Concrete policies (none/static/periodic/capacity/co_mitigation) live
in policies.py; see its docstring for how remap (config-driven, structural) and bursts
(policy-driven) are split.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

import torch

from neurofault.config import ExperimentConfig


@dataclass
class WaveContext:
    system: Any  # FaultTolerantNeuromorphic (typed Any to avoid importing torch-heavy system here)
    config: ExperimentConfig
    wave_index: int
    density: float
    device: torch.device
    testloader: Any
    canary: Any  # CanarySplit
    fisher: Any | None  # FisherTracker loaded from the checkpoint sidecar, or None
    # Per-layer EFFECTIVE stuck mask (stuck & ~remapped) as it was just BEFORE this wave's
    # injection (clones), so a policy can compute the new-fault set
    # N = effective_stuck_now & ~prev_stuck (re-failed spares count as new).
    prev_stuck: dict[str, torch.Tensor]
    test_acc_before: float
    canary_acc_before: float
    prior_events: list[dict] = field(default_factory=list)
    # Canary accuracy of the clean deployed network before wave 0. The co_mitigation gate's
    # reference for wave 0 (later waves reference the previous event's canary_acc_after).
    baseline_canary_acc: float | None = None
    # Explicit gate reference, set by the CONTINUOUS regime: the network keeps training
    # between waves, so "damage this wave did" is measured against the canary right before
    # injection, not against the previous wave's post-policy canary. None (deployment) =
    # the previous event's canary_acc_after / baseline_canary_acc as before.
    gate_reference: float | None = None


class Policy(Protocol):
    def on_wave(self, ctx: WaveContext) -> dict:
        """Act on the just-injected wave. Returns event fields to merge, e.g.
        {"burst_batches": int, "remapped": int, ...}."""


def build_policy(config: ExperimentConfig) -> Policy:
    # Imported here: policies.py imports WaveContext from this module.
    from neurofault.retraining.policies import build_policy_for

    return build_policy_for(config)
