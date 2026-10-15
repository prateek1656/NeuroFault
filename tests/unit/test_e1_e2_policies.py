"""E1 (liu_oneshot) and E2 (bn_recal_only) policies and configs. Stub systems, no CrossSim needed."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from neurofault.config import ExperimentConfig, FaultConfig, RetrainingConfig, load_config
from neurofault.retraining import policies as P

CONFIGS = Path(__file__).resolve().parents[2] / "experiments" / "configs"


def _burst_result(n):
    return SimpleNamespace(
        batches=n, optimizer_steps=n, stop_reason="fixed", canary_trace=[[n, 1.0]]
    )


def _ctx(cfg, wave, log, unremapped=0):
    def remap_now(**kw):
        log.append(("remap", kw.get("order"), kw.get("budget")))
        return {"a": 7}

    system = SimpleNamespace(
        fault_map_source="oracle",
        model="model",
        remap_now=remap_now,
        unremapped_effective_stuck=lambda: {"a": unremapped},
        unremapped_decision_stuck=lambda: {"a": unremapped},
    )
    return SimpleNamespace(
        system=system, config=cfg, wave_index=wave, device="cpu",
        canary=SimpleNamespace(burst_loader="BL", canary_loader="CL"),
        fisher=None, prior_events=[], baseline_canary_acc=90.0, gate_reference=None,
    )  # fmt: skip


def _cfg(n_waves=3, **retr):
    return SimpleNamespace(
        retraining=RetrainingConfig(policy="liu_oneshot", burst_batches=11, burst_lr=0.5, **retr),
        fault=SimpleNamespace(wave_densities=[0.1] * n_waves),
        mitigation=SimpleNamespace(enable_remapping=True),
    )


@pytest.fixture
def log(monkeypatch):
    calls = []

    def fake_burst(system, loader, **kw):
        calls.append(("burst", kw["lr"], kw["n_batches"]))
        return _burst_result(kw["n_batches"])

    monkeypatch.setattr(P, "run_burst", fake_burst)
    return calls


@pytest.mark.parametrize("at,wave", [("last", 2), ("first", 0)])
def test_liu_one_burst_in_whole_lifetime(log, at, wave):
    cfg = _cfg(liu_retrain_at=at, remap_order="saliency")
    pol = P.build_policy_for(cfg)
    assert isinstance(pol, P.LiuOneShotPolicy)
    events = [pol.on_wave(_ctx(cfg, k, log)) for k in range(3)]
    bursts = [e["burst_batches"] for e in events]
    assert bursts == [11 if k == wave else 0 for k in range(3)]
    assert [e["liu_retrain_wave"] for e in events] == [k == wave for k in range(3)]
    assert sum(1 for c in log if c[0] == "burst") == 1
    # every wave remaps greedily (no budget), in the configured order
    assert [c for c in log if c[0] == "remap"] == [("remap", "saliency", None)] * 3
    assert events[wave]["stop_reason"] == "fixed" and events[wave]["optimizer_steps"] == 11
    assert all(e["remapped_this_wave"] == 7 for e in events)
    assert "gate_drop" not in events[0] and "bn_recal_batches_used" not in events[0]  # no D parts


def test_liu_order_within_retrain_wave(log):
    pol = P.LiuOneShotPolicy()
    cfg = _cfg(liu_retrain_at="first", liu_remap_after_retrain=True)
    pol.on_wave(_ctx(cfg, 0, log))
    assert [c[0] for c in log] == ["burst", "remap"]  # Liu's evaluated order: retrain, then remap
    log.clear()
    cfg = _cfg(liu_retrain_at="first", liu_remap_after_retrain=False)
    pol.on_wave(_ctx(cfg, 0, log))
    assert [c[0] for c in log] == ["remap", "burst"]


def test_liu_uses_expected_waves_when_set(log):
    cfg = _cfg(n_waves=3, expected_waves=5)
    pol = P.LiuOneShotPolicy()
    assert pol.on_wave(_ctx(cfg, 2, log))["burst_batches"] == 0
    assert pol.on_wave(_ctx(cfg, 4, log))["burst_batches"] == 11


def test_liu_reports_unremapped(log):
    cfg = _cfg()
    f = P.LiuOneShotPolicy().on_wave(_ctx(cfg, 1, log, unremapped=5))
    assert f["unremapped_effective_stuck"] == 5


def test_bn_recal_only_never_trains(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("bn_recal_only must not run a burst")

    seen = {}
    monkeypatch.setattr(P, "run_burst", boom)
    monkeypatch.setattr(P, "recalibrate_bn", lambda m, l, n, d: seen.update(n=n, loader=l) or n)
    monkeypatch.setattr(P, "evaluate_accuracy", lambda *a, **k: 77.0)
    cfg = SimpleNamespace(
        retraining=RetrainingConfig(
            policy="bn_recal_only", bn_recal_batches=20, remap_order="random"
        ),
        mitigation=SimpleNamespace(enable_remapping=True),
    )
    pol = P.build_policy_for(cfg)
    assert isinstance(pol, P.BnRecalOnlyPolicy)
    log = []
    f = pol.on_wave(_ctx(cfg, 0, log))
    assert log == [("remap", "random", None)]
    assert seen == {"n": 20, "loader": "BL"}
    assert f["burst_batches"] == f["optimizer_steps"] == 0 and f["stop_reason"] == "no_burst"
    assert f["bn_recal_batches_used"] == 20 and f["canary_after_recal"] == 77.0
    assert f["remapped_this_wave"] == 7


def test_config_validation():
    assert (
        RetrainingConfig().liu_retrain_at == "last" and RetrainingConfig().liu_remap_after_retrain
    )
    with pytest.raises(ValueError):
        RetrainingConfig(liu_retrain_at="middle")
    with pytest.raises(ValueError):
        RetrainingConfig(policy="liu_oneshot", burst_batches=0)
    RetrainingConfig(policy="bn_recal_only")
    with pytest.raises(ValueError, match="enable_remapping"):
        ExperimentConfig(
            name="x",
            regime="deployment",
            pre_deploy_checkpoint="x.pt",
            fault=FaultConfig(wave_densities=[0.1]),
            retraining=RetrainingConfig(policy="bn_recal_only"),
        )


@pytest.mark.parametrize(
    "name", ["resnet9_e1_liu_last", "resnet9_e1_liu_first", "resnet9_e2_bn_recal_only"]
)
def test_shipped_configs_load(name):
    c = load_config(CONFIGS / f"{name}.yaml")
    assert c.regime == "deployment" and c.fault.wave_densities == [0.1, 0.1, 0.1]
    base = load_config(CONFIGS / "resnet9_final_r1.yaml")
    # same lifetime, crossbar and data as the existing arms
    assert (c.fault, c.crossbar, c.train_samples, c.test_samples) == (
        base.fault, base.crossbar, base.train_samples, base.test_samples)  # fmt: skip
    if name.startswith("resnet9_e1"):
        r = c.retraining
        assert (
            r.policy == "liu_oneshot" and r.remap_order == "saliency" and r.remap_pacing == "greedy"
        )
        assert r.burst_lr == 0.5 and r.priors_path is None
    else:
        assert c.retraining.policy == "bn_recal_only" and c.retraining.remap_order == "random"
        assert c.retraining.reference_mode == base.retraining.reference_mode == "frozen"
