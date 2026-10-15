"""Deployment regime end to end on a tiny CPU config: a real pre-deploy checkpoint is
produced in-test (float pretrain + analog epoch + Fisher), then the real lifetime loop
runs twice. No mocks at the seam under test."""

from __future__ import annotations

import json
from dataclasses import replace

import pytest
import torch

pytest.importorskip("simulator", reason="CrossSim not installed (uv sync --extra crosssim)")

from experiments.run import NOMINAL_INTER_WAVE_BATCHES, run_deployment, run_predeploy
from neurofault.config import ExperimentConfig, FaultConfig, RetrainingConfig

_TINY = {
    "simulator": "crosssim",
    "seed": 11,
    "dataset": "cifar10",
    "model_arch": "tiny_cnn",  # loop bookkeeping, not network fidelity
    "epochs": 2,
    "float_pretrain_epochs": 1,
    "num_batches": 2,
    "batch_size": 8,
    "train_samples": 64,
    "test_samples": 16,
    "eval_interval": 1,
}


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory):
    out = tmp_path_factory.mktemp("predeploy")
    config = ExperimentConfig(
        name="predeploy_tiny",
        regime="deployment",
        save_pre_deploy_checkpoint=True,
        retraining=RetrainingConfig(canary_size=16),
        **_TINY,
    )
    run_predeploy(config, out)
    return out / "pre_deploy.pt"


def _deploy_config(checkpoint, name="deploy_tiny") -> ExperimentConfig:
    return ExperimentConfig(
        name=name,
        regime="deployment",
        pre_deploy_checkpoint=str(checkpoint),
        fault=FaultConfig(wave_densities=[0.05, 0.1, 0.05]),
        retraining=RetrainingConfig(canary_size=16),
        **_TINY,
    )


def test_predeploy_checkpoint_contents(checkpoint):
    ckpt = torch.load(checkpoint, weights_only=False)
    assert set(ckpt) == {"model_state_dict", "fisher_state", "config_hash", "final_test_acc"}
    assert ckpt["fisher_state"]["num_updates"] == 2  # 1 analog epoch x 2 batches
    assert len(ckpt["config_hash"]) == 64
    # Loadable into a plain factory-built model (strict) - run_deployment relies on this.
    from neurofault.models.cnn import build_model

    build_model(_deploy_config(checkpoint)).load_state_dict(ckpt["model_state_dict"], strict=True)


def test_lifetime_is_reproducible_inference_only_and_writes_events(
    checkpoint, tmp_path, monkeypatch
):
    config = _deploy_config(checkpoint)

    step_calls = {"n": 0}
    init_calls = {"n": 0}
    real_init = torch.optim.Optimizer.__init__

    def counting_init(self, *a, **k):
        init_calls["n"] += 1
        return real_init(self, *a, **k)

    def counting_step(name):
        orig = getattr(torch.optim, name).step

        def step(self, *a, **k):
            step_calls["n"] += 1
            return orig(self, *a, **k)

        monkeypatch.setattr(getattr(torch.optim, name), "step", step)

    monkeypatch.setattr(torch.optim.Optimizer, "__init__", counting_init)
    counting_step("SGD")
    counting_step("Adam")

    out_a, out_b = tmp_path / "a", tmp_path / "b"
    out_a.mkdir()
    out_b.mkdir()
    run_deployment(config, out_a)
    run_deployment(replace(config, name="deploy_tiny_b"), out_b)

    assert step_calls["n"] == 0
    assert init_calls["n"] == 0

    ev_a = json.loads((out_a / "events.json").read_text())
    ev_b = json.loads((out_b / "events.json").read_text())

    assert ev_a["complete"] is True
    assert ev_a["nominal_inter_wave_batches"] == NOMINAL_INTER_WAVE_BATCHES == 5200
    assert ev_a["policy"] == "none"
    assert [e["wave"] for e in ev_a["events"]] == [0, 1, 2]
    required = {
        "wave", "density", "nominal_batch_equivalent", "fault_map_hash", "newly_stuck",
        "test_acc_before", "test_acc_after", "canary_acc_before", "canary_acc_after",
        "burst_batches", "remapped", "wall_seconds",
    }  # fmt: skip
    for e, density in zip(ev_a["events"], [0.05, 0.1, 0.05]):
        assert required <= set(e)
        assert e["density"] == density
        assert e["burst_batches"] == 0 and e["remapped"] == 0
        assert e["newly_stuck"] > 0
        assert e["test_acc_before"] == e["test_acc_after"]  # no-op policy
        assert e["canary_acc_before"] == e["canary_acc_after"]
    for key in ("final_test_acc", "lifetime_mean_test_acc", "worst_dip_test_acc"):
        assert key in ev_a["summary"]

    # Identical fault maps across runs (dedicated per-wave generators), and across waves
    # they differ (not a repeated map), and accumulate monotonically in count.
    hashes_a = [e["fault_map_hash"] for e in ev_a["events"]]
    assert hashes_a == [e["fault_map_hash"] for e in ev_b["events"]]
    assert len({tuple(sorted(h.items())) for h in hashes_a}) == 3
    assert [e["test_acc_after"] for e in ev_a["events"]] == [
        e["test_acc_after"] for e in ev_b["events"]
    ]
    for f in ("metrics.json", "layer_health.json", "events.json"):
        assert (out_a / f).exists()
