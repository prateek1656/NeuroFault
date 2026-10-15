"""Continuous-regime retraining policies (the C/D "bridge" arms) on a tiny CPU config
(tiny_cnn). Policy "none" must stay byte-identical to the pre-hook code (golden generated
from the pre-change commit); capacity/co_mitigation must actually inject, remap, burst and
write events.json."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

pytest.importorskip("simulator", reason="CrossSim not installed (uv sync --extra crosssim)")

from experiments.run import run_experiment
from neurofault.config import (
    CrossbarConfig,
    ExperimentConfig,
    FaultConfig,
    MitigationConfig,
    RetrainingConfig,
)

GOLDEN = Path(__file__).resolve().parent.parent / "golden"
_BASE = {
    "simulator": "crosssim",
    "seed": 5,
    "dataset": "cifar10",
    "model_arch": "tiny_cnn",
    "epochs": 4,
    "float_pretrain_epochs": 1,
    "num_batches": 4,
    "batch_size": 8,
    "train_samples": 64,
    "test_samples": 16,
    "eval_interval": 1,
}


_POLICY_MITIGATION = MitigationConfig(
    enable_soft_mitigation=False,
    enable_remapping=True,
    enable_layer_reset=False,
    enable_predictive=False,
)


def _none_config() -> ExperimentConfig:
    return ExperimentConfig(
        name="tiny_none",
        fault=FaultConfig(density=0.1, injection_interval=6),
        mitigation=MitigationConfig(enable_remapping=True, health_check_interval=2),
        **_BASE,
    )


def _policy_config(policy: str, **retraining) -> ExperimentConfig:
    return ExperimentConfig(
        name=f"tiny_{policy}",
        fault=FaultConfig(density=0.1, injection_interval=6),
        mitigation=_POLICY_MITIGATION,
        # tiny spare pool so a 10% wave exceeds capacity and the capacity policy bursts
        crossbar=CrossbarConfig(redundancy_scheme="idealized_per_cell", redundancy_factor=0.01),
        retraining=RetrainingConfig(
            policy=policy,
            reference_mode="living",
            canary_size=16,
            burst_lr=1e-3,
            burst_batches=3,
            bn_recal_batches=2,
            patience_eval_every=2,
            burst_t_max=6,
            **retraining,
        ),
        **_BASE,
    )


def test_policy_none_is_byte_equal_to_pre_hook_golden():
    metrics = run_experiment(_none_config())
    metrics.pop("epoch_seconds")  # wall-clock
    assert (
        json.dumps(metrics, indent=2) == (GOLDEN / "tiny_continuous_none_metrics.json").read_text()
    )
    assert metrics["faults"][-1] > 0


def _run(cfg, tmp_path):
    tmp_path.mkdir(parents=True, exist_ok=True)
    metrics = run_experiment(cfg, tmp_path)
    return metrics, json.loads((tmp_path / "events.json").read_text())


def test_capacity_policy_injects_remaps_and_bursts(tmp_path):
    metrics, ev = _run(_policy_config("capacity", remap_order="random"), tmp_path)
    assert ev["regime"] == "continuous" and ev["policy"] == "capacity" and ev["complete"]
    # analog batches 4..15 (global): waves at global batch 6 and 12
    assert [e["global_batch"] for e in ev["events"]] == [6, 12]
    assert all(e["newly_stuck"] > 0 for e in ev["events"])
    assert ev["summary"]["total_burst_batches"] > 0
    for e in ev["events"]:
        assert e["burst_batches"] == e["optimizer_steps"]
    assert ev["summary"]["total_remapped"] > 0
    assert metrics["faults"][-1] > 0
    assert len(metrics["loss"]) == len(metrics["accuracy"])  # metrics shapes unchanged


def test_co_mitigation_runs_and_shares_fault_maps_with_capacity(tmp_path):
    _, d = _run(
        _policy_config(
            "co_mitigation",
            remap_order="saliency",
            remap_pacing="paced",
            expected_waves=2,
            canary_gate_delta=0.0,
        ),
        tmp_path / "d",
    )
    _, c = _run(_policy_config("capacity", remap_order="random"), tmp_path / "c")
    assert d["complete"] and len(d["events"]) == 2
    assert all("burst_skipped" in e and "rho" in e for e in d["events"])
    # dedicated per-wave generator: identical injected maps whatever the policy did
    assert d["events"][0]["fault_map_hash"] == c["events"][0]["fault_map_hash"]


def test_unsupported_continuous_combos_raise_at_load_time():
    base = _policy_config("capacity")
    with pytest.raises(ValueError, match="not wired into the continuous"):
        replace(base, retraining=replace(base.retraining, policy="periodic"))
    with pytest.raises(ValueError, match="density"):
        replace(base, secondary_fault=FaultConfig(failure_model="drift", injection_interval=6))
    with pytest.raises(ValueError, match="must be False"):
        replace(base, mitigation=MitigationConfig(enable_remapping=True, enable_layer_reset=True))
    with pytest.raises(ValueError, match="freeze"):
        replace(base, retraining=replace(base.retraining, freeze_stuck_weights=False))


def test_co_mitigation_burst_fires_when_gate_trips(tmp_path, monkeypatch):
    # An untrained tiny net sits at chance, so the canary never drops and the gate would
    # always skip. Force the post-recal canary reading to 0 to exercise the burst path.
    import neurofault.retraining.policies as pol

    monkeypatch.setattr(pol, "evaluate_accuracy", lambda *a, **k: 0.0)
    _, ev = _run(
        _policy_config(
            "co_mitigation", remap_order="saliency", remap_pacing="paced", expected_waves=2
        ),
        tmp_path,
    )
    assert ev["complete"] and all(e["burst_skipped"] is False for e in ev["events"])
    assert ev["summary"]["total_burst_batches"] > 0
    for e in ev["events"]:
        assert e["burst_batches"] == e["optimizer_steps"] > 0
