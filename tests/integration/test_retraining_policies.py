"""burst executor, BN recalibration and deployment policies (R0/R1/A/B/C/C')
through the real orchestrator on a tiny CPU config: a real pre-deploy checkpoint, real
CrossSim layers, real optimizer.step() + synchronize(). Optimizer steps are counted by
wrapping (not replacing) the real step()."""

from __future__ import annotations

from dataclasses import asdict

import pytest
import torch

pytest.importorskip("simulator", reason="CrossSim not installed (uv sync --extra crosssim)")

from experiments.run import _build_system, run_deployment, run_predeploy
from neurofault.config import ExperimentConfig, FaultConfig, MitigationConfig, RetrainingConfig
from neurofault.data import load_datasets
from neurofault.models.cnn import build_model
from neurofault.retraining.bn_recal import recalibrate_bn
from neurofault.retraining.burst import run_burst
from neurofault.retraining.canary import build_canary
from neurofault.retraining.recovery_model import PatienceStopper
from neurofault.validation.results_validator import Level, overall_verdict, validate

_TINY = {
    "simulator": "crosssim",
    "seed": 11,
    "dataset": "cifar10",
    "model_arch": "tiny_cnn",  # policy/burst logic, not network fidelity
    "epochs": 2,
    "float_pretrain_epochs": 1,
    "num_batches": 2,
    "batch_size": 8,
    "train_samples": 64,
    "test_samples": 16,
    "eval_interval": 1,
}
_WAVES = [0.05, 0.1, 0.05]
_BURST = 3


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory):
    out = tmp_path_factory.mktemp("predeploy_ws3")
    config = ExperimentConfig(
        name="predeploy_tiny_ws3",
        regime="deployment",
        save_pre_deploy_checkpoint=True,
        retraining=RetrainingConfig(canary_size=16),
        **_TINY,
    )
    run_predeploy(config, out)
    return out / "pre_deploy.pt"


def _config(checkpoint, *, policy="none", remap=False, rf=0.15, reference_mode="frozen",
            order="random", name="ws3") -> ExperimentConfig:  # fmt: skip
    config = ExperimentConfig(
        name=name,
        regime="deployment",
        pre_deploy_checkpoint=str(checkpoint),
        fault=FaultConfig(wave_densities=_WAVES),
        mitigation=MitigationConfig(
            enable_soft_mitigation=False,
            enable_remapping=remap,
            enable_layer_reset=False,
            enable_predictive=False,
        ),
        retraining=RetrainingConfig(
            policy=policy,
            canary_size=16,
            burst_batches=_BURST,
            burst_lr=1e-3,
            patience_eval_every=2,
            reference_mode=reference_mode,
            remap_order=order,
        ),
        **_TINY,
    )
    config.crossbar.redundancy_factor = rf
    return config


class _StepCounter:
    def __init__(self, monkeypatch):
        self.n = 0
        for cls in (torch.optim.SGD, torch.optim.Adam):  # whichever the config selects
            monkeypatch.setattr(cls, "step", self._wrap(cls.step))

    def _wrap(self, orig):
        def step(opt, *a, **k):
            self.n += 1
            return orig(opt, *a, **k)

        return step


_RUNS: dict = {}


def _arm(checkpoint, tmp_path_factory, **kw):
    """Run (once per distinct arm, cached for the module) a real deployment lifetime and
    return (metrics, events, optimizer_steps_actually_taken). The step counter wraps the
    real optimizer.step()."""
    key = tuple(sorted(kw.items()))
    if key not in _RUNS:
        name = "arm" + str(len(_RUNS))
        config = _config(checkpoint, name=name, **kw)
        out = tmp_path_factory.mktemp(name)
        with pytest.MonkeyPatch.context() as mp:
            counter = _StepCounter(mp)
            metrics = run_deployment(config, out)
        _RUNS[key] = (config, metrics, metrics["events"]["events"], counter.n)
    return _RUNS[key]


def test_r0_takes_zero_optimizer_steps(checkpoint, tmp_path_factory):
    _, _, events, steps = _arm(checkpoint, tmp_path_factory)
    assert steps == 0
    for e in events:
        assert e["optimizer_steps"] == e["burst_batches"] == 0
        assert e["remapped_this_wave"] == 0
        assert e["stop_reason"] == "no_burst"


def test_r1_remaps_every_wave_without_retraining(checkpoint, tmp_path_factory):
    _, metrics, events, steps = _arm(checkpoint, tmp_path_factory, remap=True)
    assert steps == 0
    assert all(e["remapped_this_wave"] > 0 for e in events)
    assert all(e["remapped"] == e["remapped_this_wave"] for e in events)  # system.stats agrees
    assert metrics["remappings"][-1] == sum(e["remapped_this_wave"] for e in events)
    assert all(e["burst_batches"] == 0 for e in events)


def test_remap_does_not_change_the_injected_fault_maps(checkpoint, tmp_path_factory):
    _, _, r0, _ = _arm(checkpoint, tmp_path_factory)
    _, _, r1, _ = _arm(checkpoint, tmp_path_factory, remap=True)
    assert [e["fault_map_hash"] for e in r0] == [e["fault_map_hash"] for e in r1]


def test_periodic_bursts_every_wave_static_only_wave_zero(checkpoint, tmp_path_factory):
    _, _, events, steps = _arm(checkpoint, tmp_path_factory, policy="periodic")
    assert steps == _BURST * len(_WAVES)
    for e in events:
        assert e["burst_batches"] == e["optimizer_steps"] == _BURST
        assert e["stop_reason"] == "fixed"
        assert [t[0] for t in e["canary_trace"]] == [2]  # eval_every=2 within 3 batches

    _, _, events, steps = _arm(checkpoint, tmp_path_factory, policy="static")
    assert steps == _BURST
    assert [e["optimizer_steps"] for e in events] == [_BURST, 0, 0]


def test_capacity_bursts_exactly_when_the_wave_exceeded_capacity(checkpoint, tmp_path_factory):
    cap = {"policy": "capacity", "remap": True}

    # tiny capacity: spares run out in every wave -> a burst every wave
    _, _, events, _ = _arm(checkpoint, tmp_path_factory, rf=0.001, **cap)
    assert all(e["unremapped_effective_stuck"] > 0 for e in events)
    assert [e["optimizer_steps"] for e in events] == [_BURST] * 3

    # huge capacity: every stuck cell bypassed -> no burst at all
    _, _, events, steps = _arm(checkpoint, tmp_path_factory, rf=0.9, **cap)
    assert all(e["unremapped_effective_stuck"] == 0 for e in events)
    assert steps == 0

    # in between: a burst iff cells remain unremapped (whatever the pattern turns out to be)
    _, _, events, _ = _arm(checkpoint, tmp_path_factory, rf=0.08, **cap)
    for e in events:
        assert (e["optimizer_steps"] > 0) == (e["unremapped_effective_stuck"] > 0)
    assert any(e["optimizer_steps"] > 0 for e in events)


def test_capacity_living_matches_frozen_decisions(checkpoint, tmp_path_factory):
    kw = {"policy": "capacity", "remap": True, "rf": 0.08}
    _, _, frozen, _ = _arm(checkpoint, tmp_path_factory, **kw)
    _, _, living, _ = _arm(checkpoint, tmp_path_factory, reference_mode="living", **kw)
    # C vs C' differ ONLY by reference_mode: same fault maps, same remap counts, same triggers
    assert [e["fault_map_hash"] for e in frozen] == [e["fault_map_hash"] for e in living]
    assert [e["remapped_this_wave"] for e in frozen] == [e["remapped_this_wave"] for e in living]
    assert [e["optimizer_steps"] for e in frozen] == [e["optimizer_steps"] for e in living]


def test_events_validate_and_deployment_loss_sentinel_is_skipped(checkpoint, tmp_path_factory):
    config, metrics, events, _ = _arm(checkpoint, tmp_path_factory, policy="periodic", remap=True)
    metrics = {k: v for k, v in metrics.items() if k not in ("events", "layer_health")}
    full_events = _RUNS[tuple(sorted({"policy": "periodic", "remap": True}.items()))][1]["events"]
    required = {
        "burst_batches", "stop_reason", "canary_trace", "remapped_this_wave",
        "unremapped_effective_stuck", "optimizer_steps",
    }  # fmt: skip
    assert all(required <= set(e) for e in events)

    results = validate(metrics, asdict(config), results_name=config.name, events=full_events)
    assert overall_verdict(results) != Level.FAIL
    names = {r.name for r in results}
    assert not any(n.startswith("qualitative.loss") or n == "outlier.loss_jump" for n in names)
    assert all(r.level == Level.PASS for r in results if r.name.startswith("deployment."))

    # the validator really fails on a broken record
    bad = {**full_events, "events": [{**full_events["events"][0], "optimizer_steps": 99}]}
    assert any(r.level == Level.FAIL for r in validate(metrics, asdict(config), events=bad))
    assert any(
        r.name == "deployment.events_present" and r.level == Level.FAIL
        for r in validate(metrics, asdict(config), events=None)
    )


def test_capacity_policy_config_validation(checkpoint):
    with pytest.raises(ValueError, match="enable_remapping"):
        _config(checkpoint, policy="capacity", remap=False)


# --- burst / BN recalibration unit-level behavior ------------------------------------


def _system_and_canary(checkpoint, config):
    trainloader, _ = load_datasets(config)
    base = build_model(config)
    base.load_state_dict(torch.load(checkpoint, weights_only=False)["model_state_dict"])
    return _build_system(config, base, trainloader, torch.device("cpu")), build_canary(config)


def test_burst_counts_steps_exactly_and_canary_is_disjoint(checkpoint, monkeypatch):
    config = _config(checkpoint, policy="periodic")
    system, canary = _system_and_canary(checkpoint, config)
    assert set(canary.canary_indices).isdisjoint(canary.burst_indices)
    counter = _StepCounter(monkeypatch)

    n = len(canary.burst_loader) * 2 + 1  # forces the loader to be cycled
    result = run_burst(
        system, canary.burst_loader, lr=1e-3, n_batches=n, canary=canary.canary_loader,
        eval_every=4, config=config, device=torch.device("cpu"),
    )  # fmt: skip
    assert result.batches == result.optimizer_steps == counter.n == n
    assert [t[0] for t in result.canary_trace] == list(range(4, n + 1, 4))
    assert not system.model.training  # handed back in eval mode

    with pytest.raises(ValueError, match="exactly one"):
        run_burst(system, canary.burst_loader, lr=1e-3, config=config, device="cpu")


def test_burst_with_patience_stopper_stops_on_t_max(checkpoint, monkeypatch):
    config = _config(checkpoint, policy="periodic")
    system, canary = _system_and_canary(checkpoint, config)
    counter = _StepCounter(monkeypatch)
    stopper = PatienceStopper(eval_every=2, window=50, min_gain=0.0, t_max=5)
    result = run_burst(
        system, canary.burst_loader, lr=1e-3, stopper=stopper, canary=canary.canary_loader,
        config=config, device=torch.device("cpu"),
    )  # fmt: skip
    assert result.stop_reason == "t_max"
    assert result.batches == result.optimizer_steps == counter.n == 5


def test_burst_keeps_stuck_weights_frozen_and_pinned(checkpoint):
    config = _config(checkpoint, policy="periodic")
    system, canary = _system_and_canary(checkpoint, config)
    g = torch.Generator().manual_seed(1)
    system.inject_faults(generator=g)
    system.apply_pins()
    before = {n: system._weight_matrix(h).clone() for n, h in system.handles.items()}
    run_burst(
        system, canary.burst_loader, lr=1e-2, n_batches=4, config=config, device=torch.device("cpu")
    )
    for name, handle in system.handles.items():
        stuck = system.monitors[name].effective_stuck
        assert torch.equal(system._weight_matrix(handle)[stuck], before[name][stuck])


def test_bn_recalibration_leaves_analog_matrices_identical(checkpoint):
    config = _config(checkpoint, policy="periodic")
    system, canary = _system_and_canary(checkpoint, config)
    system.model.eval()
    matrices = {n: h.accessor.read().clone() for n, h in system.handles.items()}
    weights = {n: p.detach().clone() for n, p in system.model.named_parameters()}
    bns = [m for m in system.model.modules() if isinstance(m, torch.nn.BatchNorm2d)]
    assert bns
    stats_before = [bn.running_mean.clone() for bn in bns]
    momenta = [bn.momentum for bn in bns]

    used = recalibrate_bn(system.model, canary.burst_loader, 2, torch.device("cpu"))

    assert used == 2
    assert any(not torch.equal(b, bn.running_mean) for b, bn in zip(stats_before, bns))
    for n, h in system.handles.items():
        assert torch.equal(h.accessor.read(), matrices[n])
    for n, p in system.model.named_parameters():
        assert torch.equal(p, weights[n])
    assert [bn.momentum for bn in bns] == momenta
    assert not any(m.training for m in system.model.modules())  # modes restored (all eval)
    assert recalibrate_bn(system.model, canary.burst_loader, 0, "cpu") == 0
