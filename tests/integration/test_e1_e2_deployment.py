"""E1 / E2 policies through the real orchestrator on a tiny CPU config (real CrossSim layers,
real optimizer steps). Checks: E1 takes exactly one burst in the lifetime, E2 takes none and
changes the BN statistics, and the fault maps are identical to the R1 arm's."""

from __future__ import annotations

import pytest
import torch

pytest.importorskip("simulator", reason="CrossSim not installed (uv sync --extra crosssim)")

from experiments.run import run_deployment, run_predeploy
from neurofault.config import ExperimentConfig, FaultConfig, MitigationConfig, RetrainingConfig

_TINY = {
    "simulator": "crosssim", "seed": 11, "dataset": "cifar10", "model_arch": "tiny_cnn",
    "epochs": 2, "float_pretrain_epochs": 1, "num_batches": 2, "batch_size": 8,
    "train_samples": 64, "test_samples": 16, "eval_interval": 1,
}  # fmt: skip
_WAVES = [0.05, 0.1, 0.05]
_BURST = 3


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory):
    out = tmp_path_factory.mktemp("predeploy_e1e2")
    config = ExperimentConfig(
        name="predeploy_tiny_e1e2", regime="deployment", save_pre_deploy_checkpoint=True,
        retraining=RetrainingConfig(canary_size=16), **_TINY)  # fmt: skip
    run_predeploy(config, out)
    return out / "pre_deploy.pt"


def _run(checkpoint, tmp_path_factory, name, **retr):
    config = ExperimentConfig(
        name=name,
        regime="deployment",
        pre_deploy_checkpoint=str(checkpoint),
        fault=FaultConfig(wave_densities=_WAVES),
        mitigation=MitigationConfig(
            enable_soft_mitigation=False,
            enable_remapping=True,
            enable_layer_reset=False,
            enable_predictive=False,
        ),
        retraining=RetrainingConfig(
            canary_size=16, burst_batches=_BURST, burst_lr=1e-3, patience_eval_every=2, **retr
        ),
        **_TINY,
    )
    config.crossbar.redundancy_factor = 0.08
    return run_deployment(config, tmp_path_factory.mktemp(name))["events"]["events"]


@pytest.mark.parametrize("at,wave", [("last", 2), ("first", 0)])
def test_e1_single_burst(checkpoint, tmp_path_factory, at, wave):
    ev = _run(checkpoint, tmp_path_factory, f"liu_{at}", policy="liu_oneshot", reference_mode="living",
              remap_order="saliency", liu_retrain_at=at)  # fmt: skip
    assert [e["optimizer_steps"] for e in ev] == [_BURST if k == wave else 0 for k in range(3)]
    assert [e["burst_batches"] for e in ev] == [e["optimizer_steps"] for e in ev]
    assert all(e["remapped_this_wave"] > 0 for e in ev[:1])  # later waves may find the pool empty
    r1 = _run(checkpoint, tmp_path_factory, "r1_for_liu", policy="none", remap_order="saliency",
              reference_mode="living")  # fmt: skip
    assert [e["fault_map_hash"] for e in ev] == [e["fault_map_hash"] for e in r1]


def test_e2_no_gradient_steps_and_bn_stats_move(checkpoint, tmp_path_factory, monkeypatch):
    steps = []
    for cls in (torch.optim.SGD, torch.optim.Adam):
        orig = cls.step
        monkeypatch.setattr(
            cls, "step", lambda o, *a, _o=orig, **k: steps.append(1) or _o(o, *a, **k)
        )
    ev = _run(checkpoint, tmp_path_factory, "bn_only", policy="bn_recal_only", bn_recal_batches=2)
    r1 = _run(checkpoint, tmp_path_factory, "r1_for_bn", policy="none")
    assert steps == []
    assert all(e["bn_recal_batches_used"] == 2 and e["burst_batches"] == 0 for e in ev)
    assert [e["fault_map_hash"] for e in ev] == [e["fault_map_hash"] for e in r1]
    assert [e["remapped_this_wave"] for e in ev] == [e["remapped_this_wave"] for e in r1]
