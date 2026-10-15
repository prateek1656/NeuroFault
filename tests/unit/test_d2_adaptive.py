"""D2 switches: canary_sigma, config
validation, run_burst anneal / restore_best, and the policy gate threshold. Uses a tiny
Linear model and a stub system, so no CrossSim is needed."""

from __future__ import annotations

import itertools
import math
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from neurofault.config import ExperimentConfig, RetrainingConfig
from neurofault.retraining import burst as burst_mod
from neurofault.retraining import policies as policies_mod
from neurofault.retraining.burst import run_burst
from neurofault.retraining.canary import canary_sigma
from neurofault.retraining.policies import (
    CapacityOrCanaryPolicy,
    CoMitigationPolicy,
    build_policy_for,
)
from neurofault.retraining.recovery_model import PatienceStopper


class StubSystem:
    def __init__(self, model):
        self.model = model
        self.syncs = 0

    def synchronize(self):
        self.syncs += 1


def _model():
    torch.manual_seed(0)
    return torch.nn.Linear(4, 3)


def _loader(n_batches=6):
    g = torch.Generator().manual_seed(1)
    return [
        (torch.randn(8, 4, generator=g), torch.randint(0, 3, (8,), generator=g))
        for _ in range(n_batches)
    ]


def _cfg(**retr):
    return ExperimentConfig(name="t", retraining=RetrainingConfig(**retr))


def _pcfg(**retr):
    """Config stand-in for the policies (full ExperimentConfig needs a deployment setup)."""
    return SimpleNamespace(
        retraining=RetrainingConfig(**retr), mitigation=SimpleNamespace(enable_remapping=False)
    )


def _reference_run_burst(system, loader, lr, n_batches, canary, eval_every, config):
    """Copy of the OLD fixed-length run_burst loop (before the D2 switches)."""
    model = system.model
    opt = burst_mod._build_burst_optimizer(model.parameters(), config, lr)
    crit = torch.nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
    trace, batches, steps = [], 0, 0
    model.train()
    done = False
    while not done:
        for x, y in loader:
            for g in opt.param_groups:
                g["lr"] = lr * 0.5 * (1.0 + math.cos(math.pi * batches / n_batches))
            opt.zero_grad()
            crit(model(x), y).backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                config.grad_clip_norm if config.grad_clip_norm > 0 else float("inf"),
            )
            opt.step()
            steps += 1
            system.synchronize()
            batches += 1
            if batches % eval_every == 0:
                trace.append([batches, burst_mod.evaluate_accuracy(model, canary, "cpu")])
            if batches >= n_batches:
                done = True
                break
    model.eval()
    return batches, steps, trace


# (d)
def test_canary_sigma_values():
    assert canary_sigma(90.0, 1024) == pytest.approx(0.9375, abs=1e-3)
    assert canary_sigma(50.0, 1024) == pytest.approx(100 * math.sqrt(0.25 / 1024))
    # clamp at both ends
    assert canary_sigma(100.0, 1024) == canary_sigma(99.0, 1024)
    assert canary_sigma(0.0, 1024) == canary_sigma(1.0, 1024)


# (a)
def test_switches_off_matches_old_path():
    cfg = _cfg()
    canary = _loader(2)
    m_new, m_old = _model(), _model()
    sys_new, sys_old = StubSystem(m_new), StubSystem(m_old)
    res = run_burst(sys_new, _loader(), lr=0.1, n_batches=10, canary=canary, eval_every=3,
                    config=cfg, device="cpu")  # fmt: skip
    b, s, trace = _reference_run_burst(sys_old, _loader(), 0.1, 10, canary, 3, cfg)
    assert (res.batches, res.optimizer_steps, res.canary_trace) == (b, s, trace)
    assert res.stop_reason == "fixed" and res.restored_best is False and res.best_canary is None
    assert sys_new.syncs == sys_old.syncs == 10
    for a, c in zip(m_new.parameters(), m_old.parameters()):
        assert torch.equal(a, c)


def test_switches_off_stopper_path_is_unchanged_by_extra_fields():
    cfg = _cfg()
    sysm = StubSystem(_model())
    stopper = PatienceStopper(2, 1, 1000.0, 40)  # min_gain huge => plateau at first full window
    res = run_burst(
        sysm, _loader(), lr=0.1, stopper=stopper, canary=_loader(2), config=cfg, device="cpu"
    )
    assert res.stop_reason == "plateau" and res.batches == 4 and sysm.syncs == 4


# (b)
def test_anneal_counts_extra_steps_and_ends_at_lr_zero(monkeypatch):
    cfg = _cfg(anneal_on_stop=True, anneal_steps=7)
    sysm = StubSystem(_model())
    seen_lrs = []
    holder = {}
    real_build = burst_mod._build_burst_optimizer

    def build(*args, **kw):
        holder["opt"] = real_build(*args, **kw)
        return holder["opt"]

    sysm.synchronize = lambda: seen_lrs.append(holder["opt"].param_groups[0]["lr"])
    monkeypatch.setattr(burst_mod, "_build_burst_optimizer", build)
    stopper = PatienceStopper(2, 1, 1000.0, 40)
    res = run_burst(
        sysm, _loader(), lr=0.1, stopper=stopper, canary=_loader(2), config=cfg, device="cpu"
    )
    assert holder["opt"].param_groups[0]["lr"] == 0.0  # the anneal ends at LR 0
    assert res.stop_reason == "plateau+anneal"
    assert res.batches == res.optimizer_steps == len(seen_lrs) == 4 + 7
    anneal_lrs = seen_lrs[4:]
    assert anneal_lrs[0] == pytest.approx(0.1 * 0.5 * (1 + math.cos(math.pi * 4 / 40)))
    assert all(a >= b for a, b in itertools.pairwise(anneal_lrs))
    assert anneal_lrs[-1] < 0.05 * anneal_lrs[0]


def test_anneal_not_applied_without_switch_or_on_fixed_burst():
    sysm = StubSystem(_model())
    res = run_burst(sysm, _loader(), lr=0.1, n_batches=5, canary=_loader(2), eval_every=5,
                    config=_cfg(anneal_on_stop=True), device="cpu")  # fmt: skip
    assert res.batches == 5 and res.stop_reason == "fixed"


# (c)
def test_restore_best_restores_earlier_state(monkeypatch):
    readings = iter([80.0, 90.0, 70.0])  # evals at batches 2, 4, 6; the last is worse
    snapshots = []

    def fake_eval(model, loader, device):
        snapshots.append({k: v.clone() for k, v in model.state_dict().items()})
        return next(readings)

    monkeypatch.setattr(burst_mod, "evaluate_accuracy", fake_eval)
    model = _model()
    sysm = StubSystem(model)
    res = run_burst(sysm, _loader(), lr=0.1, n_batches=6, canary=[None], eval_every=2,
                    config=_cfg(restore_best=True), device="cpu")  # fmt: skip
    assert res.restored_best and res.best_canary == 90.0
    assert res.batches == 6 and sysm.syncs == 6 + 1  # one extra synchronize after the restore
    for k, v in model.state_dict().items():
        assert torch.equal(v, snapshots[1][k])  # weights at the 90.0 reading


def test_restore_best_ignores_gap_smaller_than_one_sigma(monkeypatch):
    readings = iter([80.0, 90.0, 89.5])  # final is 0.5 below best; sigma at 90 / 1024 is about 0.94
    monkeypatch.setattr(burst_mod, "evaluate_accuracy", lambda *a: next(readings))
    sysm = StubSystem(_model())
    res = run_burst(sysm, _loader(), lr=0.1, n_batches=6, canary=[None], eval_every=2,
                    config=_cfg(restore_best=True), device="cpu")  # fmt: skip
    assert not res.restored_best and res.best_canary == 90.0 and sysm.syncs == 6


def test_restore_best_keeps_final_when_it_is_best(monkeypatch):
    readings = iter([70.0, 80.0, 90.0])
    monkeypatch.setattr(burst_mod, "evaluate_accuracy", lambda *a: next(readings))
    sysm = StubSystem(_model())
    res = run_burst(sysm, _loader(), lr=0.1, n_batches=6, canary=[None], eval_every=2,
                    config=_cfg(restore_best=True), device="cpu")  # fmt: skip
    assert not res.restored_best and res.best_canary == 90.0 and sysm.syncs == 6


def test_restore_best_after_anneal_evaluates_at_the_end(monkeypatch):
    # stopper fires at batch 2 (eval 1, window 1 => needs 2 obs: fires at batch 4), anneal 3 steps
    readings = iter([60.0, 60.0, 50.0])  # evals at 2, 4, then end of anneal (batch 7)
    monkeypatch.setattr(burst_mod, "evaluate_accuracy", lambda *a: next(readings))
    sysm = StubSystem(_model())
    stopper = PatienceStopper(2, 1, 1000.0, 40)
    res = run_burst(sysm, _loader(), lr=0.1, stopper=stopper, canary=[None],
                    config=_cfg(anneal_on_stop=True, anneal_steps=3, restore_best=True), device="cpu")  # fmt: skip
    assert res.batches == 7 and res.canary_trace[-1] == [7, 50.0]
    assert res.restored_best and res.best_canary == 60.0 and res.stop_reason == "plateau+anneal"


# (e)
def test_gate_threshold_with_noise_k(monkeypatch):
    monkeypatch.setattr(policies_mod, "recalibrate_bn", lambda *a, **k: 0)
    monkeypatch.setattr(policies_mod, "evaluate_accuracy", lambda *a, **k: 89.0)
    calls = []
    monkeypatch.setattr(policies_mod, "run_burst", lambda *a, **k: calls.append(k) or None)
    cfg = _pcfg(policy="co_mitigation", reference_mode="living", gate_noise_k=2.0, canary_size=1024,
               canary_gate_reference="best", noise_aware_plateau=True)  # fmt: skip
    pol = CoMitigationPolicy.__new__(CoMitigationPolicy)
    pol.params = {"canary_gate_delta": 1.0, "remap_pacing": "greedy", "patience_eval_every": 25,
                  "patience_window": 3, "patience_min_gain": 0.5, "burst_t_max": 1000,
                  "burst_lr": 0.01}  # fmt: skip
    pol.from_priors = []
    system = SimpleNamespace(
        pools={}, fault_map_source="oracle", model=None,
        remap_now=lambda **k: {}, unremapped_effective_stuck=dict, unremapped_decision_stuck=dict,
    )  # fmt: skip
    ctx = SimpleNamespace(system=system, config=cfg, wave_index=0, device="cpu",
                          canary=SimpleNamespace(burst_loader=None, canary_loader=None),
                          prior_events=[], baseline_canary_acc=90.0, gate_reference=None, fisher=None)  # fmt: skip
    # drop = 1.0 < 1.88: skipped even though old delta (1.0) would not skip (1.0 <= 1.0 skips too);
    # use drop 1.5 below instead
    monkeypatch.setattr(policies_mod, "evaluate_accuracy", lambda *a, **k: 88.5)
    f = pol.on_wave(ctx)
    assert f["gate_threshold"] == pytest.approx(2 * canary_sigma(90.0, 1024))
    assert f["gate_threshold"] == pytest.approx(1.875, abs=0.01)
    assert f["canary_sigma"] == pytest.approx(0.9375, abs=1e-3)
    assert f["gate_drop"] == pytest.approx(1.5) and f["burst_skipped"] is True and not calls


def test_noise_aware_plateau_min_gain_recorded(monkeypatch):
    monkeypatch.setattr(policies_mod, "recalibrate_bn", lambda *a, **k: 0)
    monkeypatch.setattr(policies_mod, "evaluate_accuracy", lambda *a, **k: 80.0)
    monkeypatch.setattr(policies_mod, "run_burst",
                        lambda *a, **k: SimpleNamespace(batches=1, optimizer_steps=1, stop_reason="plateau",
                                                        canary_trace=[], restored_best=False, best_canary=80.0))  # fmt: skip
    cfg = _pcfg(
        policy="co_mitigation", reference_mode="living", noise_aware_plateau=True, restore_best=True
    )
    pol = CoMitigationPolicy.__new__(CoMitigationPolicy)
    pol.params = {"canary_gate_delta": 1.0, "remap_pacing": "greedy", "patience_eval_every": 25,
                  "patience_window": 4, "patience_min_gain": 0.5, "burst_t_max": 1000, "burst_lr": 0.01}  # fmt: skip
    pol.from_priors = []
    system = SimpleNamespace(pools={}, fault_map_source="oracle", model=None, remap_now=lambda **k: {},
                             unremapped_effective_stuck=dict, unremapped_decision_stuck=dict)  # fmt: skip
    ctx = SimpleNamespace(system=system, config=cfg, wave_index=0, device="cpu",
                          canary=SimpleNamespace(burst_loader=None, canary_loader=None),
                          prior_events=[], baseline_canary_acc=95.0, gate_reference=None, fisher=None)  # fmt: skip
    f = pol.on_wave(ctx)
    assert f["plateau_min_gain"] == pytest.approx(canary_sigma(80.0, 1024) / 2.0)
    assert f["restored_best"] is False and "gate_threshold" not in f


def test_capacity_or_canary_policy(monkeypatch):
    seen = {}
    monkeypatch.setattr(policies_mod, "evaluate_accuracy", lambda *a, **k: seen["acc"])
    calls = []
    monkeypatch.setattr(policies_mod, "run_burst",
                        lambda *a, **k: calls.append(k) or SimpleNamespace(
                            batches=5, optimizer_steps=5, stop_reason="fixed", canary_trace=[]))  # fmt: skip
    cfg = _pcfg(
        policy="capacity_or_canary", gate_noise_k=2.0, burst_batches=5, reference_mode="living"
    )
    pol = build_policy_for(cfg)
    assert isinstance(pol, CapacityOrCanaryPolicy)

    def ctx(believed, prior):
        system = SimpleNamespace(fault_map_source="oracle", model=None, remap_now=lambda **k: {},
                                 unremapped_effective_stuck=lambda: {"a": believed},
                                 unremapped_decision_stuck=lambda: {"a": believed})  # fmt: skip
        return SimpleNamespace(system=system, config=cfg, wave_index=len(prior), device="cpu",
                               canary=SimpleNamespace(burst_loader=None, canary_loader=None),
                               prior_events=[{"canary_acc_after": a} for a in prior],
                               baseline_canary_acc=90.0, gate_reference=None, fisher=None)  # fmt: skip

    seen["acc"] = 89.0  # drop 1.0 < 1.875, no unremapped: no burst
    f = pol.on_wave(ctx(0, []))
    assert f["burst_skipped"] and not calls and f["gate_reference"] == 90.0
    seen["acc"] = 87.0  # drop 3.0 > 1.875: canary trigger
    f = pol.on_wave(ctx(0, []))
    assert (
        not f["burst_skipped"]
        and f["canary_trigger"]
        and not f["capacity_trigger"]
        and len(calls) == 1
    )
    assert f["burst_batches"] == 5
    seen["acc"] = 90.0  # unremapped cells: capacity trigger even with no canary drop
    f = pol.on_wave(ctx(3, []))
    assert f["capacity_trigger"] and not f["canary_trigger"] and len(calls) == 2
    # the reference is the BEST earlier canary (92), not the previous one (85)
    seen["acc"] = 89.0
    f = pol.on_wave(ctx(0, [92.0, 85.0]))
    assert f["gate_reference"] == 92.0 and f["canary_trigger"]


# (f)
def test_config_validation():
    assert (
        RetrainingConfig(policy="capacity_or_canary", gate_noise_k=2.0).policy
        == "capacity_or_canary"
    )
    with pytest.raises(ValueError):
        RetrainingConfig(policy="capacity_or_canary")  # needs gate_noise_k
    with pytest.raises(ValueError):
        RetrainingConfig(gate_noise_k=0.0)
    with pytest.raises(ValueError):
        RetrainingConfig(anneal_steps=0)
    r = RetrainingConfig()
    assert (r.gate_noise_k, r.noise_aware_plateau, r.anneal_on_stop, r.restore_best, r.anneal_steps) == (
        None, False, False, False, 50)  # fmt: skip
    assert replace(r, gate_noise_k=2.0).gate_noise_k == 2.0
