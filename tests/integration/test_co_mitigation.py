"""policy D (co_mitigation), the priors overlay, the calibration sweep and its
consumer script, all through the real orchestrator on a tiny CPU config (real CrossSim
layers, real optimizer.step() + synchronize())."""

from __future__ import annotations

import importlib.util
import json
import math
from dataclasses import asdict
from pathlib import Path

import pytest
import torch

pytest.importorskip("simulator", reason="CrossSim not installed (uv sync --extra crosssim)")

from experiments.calibration_sweep import run_sweep
from experiments.run import _build_system, _inject_wave, run_deployment, run_predeploy
from neurofault.config import ExperimentConfig, FaultConfig, MitigationConfig, RetrainingConfig
from neurofault.data import load_datasets
from neurofault.models.cnn import build_model
from neurofault.retraining.canary import build_canary
from neurofault.retraining.policies import effective_params
from neurofault.retraining.policy_hook import WaveContext, build_policy
from neurofault.validation.results_validator import Level, overall_verdict, validate

ROOT = Path(__file__).resolve().parents[2]
_TINY = {
    "simulator": "crosssim",
    "seed": 11,
    "dataset": "cifar10",
    "model_arch": "tiny_cnn",  # policy logic, not network fidelity
    "epochs": 2,
    "float_pretrain_epochs": 1,
    "num_batches": 2,
    "batch_size": 8,
    "train_samples": 64,
    "test_samples": 16,
    "eval_interval": 1,
}
_WAVES = [0.05, 0.1, 0.05]


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory):
    out = tmp_path_factory.mktemp("predeploy_ws4b")
    config = ExperimentConfig(
        name="predeploy_tiny_ws4b",
        regime="deployment",
        save_pre_deploy_checkpoint=True,
        retraining=RetrainingConfig(canary_size=16),
        **_TINY,
    )
    run_predeploy(config, out)
    return out / "pre_deploy.pt"


def _config(checkpoint, *, rf=0.06, name="d", **retr) -> ExperimentConfig:
    fields = {
        "policy": "co_mitigation",
        "reference_mode": "living",
        "remap_order": "saliency",
        "remap_pacing": "paced",
        "burst_mode": "fixed",
        "burst_batches": 3,
        "burst_lr": 1e-2,
        "patience_eval_every": 2,
        "patience_window": 1,
        "bn_recal_batches": 2,
        "canary_size": 16,
    }
    fields.update(retr)
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
        retraining=RetrainingConfig(**fields),
        **_TINY,
    )
    config.crossbar.redundancy_factor = rf
    return config


def _run(checkpoint, tmp_path, **kw):
    config = _config(checkpoint, **kw)
    metrics = run_deployment(config, tmp_path)
    return config, metrics, metrics["events"]


def _ctx(checkpoint, config, baseline_canary):
    """A real system + a wave injected exactly as run_deployment does, as a WaveContext."""
    trainloader, testloader = load_datasets(config)
    base = build_model(config)
    base.load_state_dict(torch.load(checkpoint, weights_only=False)["model_state_dict"])
    system = _build_system(config, base, trainloader, torch.device("cpu"))
    system.fisher = None
    canary = build_canary(config)
    prev = {n: m.effective_stuck.clone() for n, m in system.monitors.items()}
    _inject_wave(system, config.seed * 1000)
    system.apply_pins()
    ctx = WaveContext(
        system=system, config=config, wave_index=0, density=_WAVES[0], device=torch.device("cpu"),
        testloader=testloader, canary=canary, fisher=None, prev_stuck=prev,
        test_acc_before=0.0, canary_acc_before=0.0, baseline_canary_acc=baseline_canary,
    )  # fmt: skip
    return system, ctx


def test_paced_budgets_and_rho_sequence(checkpoint, tmp_path):
    _, _, ev = _run(checkpoint, tmp_path)
    events = ev["events"]
    assert [e["rho"] for e in events] == pytest.approx([1 / 3, 1 / 2, 1.0])
    for e in events:
        for layer, avail in e["available_capacity_by_layer"].items():
            assert e["budget_by_layer"][layer] == math.floor(avail * e["rho"])
            assert e["remapped_by_layer"][layer] <= e["budget_by_layer"][layer]
    # rf=0.06 with 5% faults vs a 2% wave-0 budget: the cap binds in some layer, so the
    # pacing really withheld spares (otherwise this test proves nothing)
    e0 = events[0]
    assert any(
        e0["remapped_by_layer"][n] == e0["budget_by_layer"][n] > 0 for n in e0["budget_by_layer"]
    )
    assert e0["unremapped_effective_stuck"] > 0
    assert e0["remapped_this_wave"] == sum(e0["remapped_by_layer"].values())
    # spares left after wave 0 are what wave 1's budget is computed from
    for layer, avail in events[1]["available_capacity_by_layer"].items():
        assert (
            avail
            == events[0]["available_capacity_by_layer"][layer] - e0["remapped_by_layer"][layer]
        )


def test_greedy_has_no_budget_and_rho_one(checkpoint, tmp_path):
    _, _, ev = _run(checkpoint, tmp_path, remap_pacing="greedy", rf=0.9)
    for e in ev["events"]:
        assert e["rho"] == 1.0 and e["budget_by_layer"] is None
        assert e["unremapped_effective_stuck"] == 0  # everything fits


def test_expected_waves_override_changes_rho(checkpoint, tmp_path):
    _, _, ev = _run(checkpoint, tmp_path, expected_waves=6)
    assert [e["rho"] for e in ev["events"]] == pytest.approx([1 / 6, 1 / 5, 1 / 4])


def test_gate_skips_burst_when_drop_within_delta_and_events_validate(checkpoint, tmp_path):
    config, metrics, ev = _run(checkpoint, tmp_path, canary_gate_delta=1000.0)
    for e in ev["events"]:
        assert e["burst_skipped"] and e["stop_reason"] == "no_burst"
        assert e["burst_batches"] == e["optimizer_steps"] == 0
    required = {
        "rho", "budget_by_layer", "remapped_by_layer", "canary_after_recal", "gate_reference",
        "gate_drop", "burst_skipped", "effective_params", "params_from_priors", "canary_trace",
        "stop_reason", "unremapped_effective_stuck",
    }  # fmt: skip
    assert all(required <= set(e) for e in ev["events"])
    # gate reference chain: baseline for wave 0, then the previous wave's post-mitigation canary
    assert ev["events"][0]["gate_reference"] == ev["baseline_canary_acc"]
    assert ev["events"][1]["gate_reference"] == ev["events"][0]["canary_acc_after"]
    for e in ev["events"]:
        assert e["gate_drop"] == e["gate_reference"] - e["canary_after_recal"]

    plain = {k: v for k, v in metrics.items() if k not in ("events", "layer_health")}
    results = validate(plain, asdict(config), results_name=config.name, events=ev)
    assert overall_verdict(results) != Level.FAIL
    assert all(r.level == Level.PASS for r in results if r.name.startswith("deployment."))


def test_gate_decision_matches_drop_through_the_lifetime(checkpoint, tmp_path):
    _, _, ev = _run(checkpoint, tmp_path, canary_gate_delta=0.0)
    for e in ev["events"]:
        assert e["burst_skipped"] == (e["gate_drop"] <= 0.0)
        assert (e["optimizer_steps"] > 0) == (not e["burst_skipped"])
        assert e["burst_batches"] == e["optimizer_steps"]


def _counting(orig, steps):
    def step(opt, *a, **k):
        steps["n"] += 1
        return orig(opt, *a, **k)

    return step


def test_forced_burst_trains_living_spares_and_adaptive_stops(checkpoint, monkeypatch):
    # baseline 100% canary => drop is large => the gate must NOT skip
    config = _config(checkpoint, remap_order="magnitude", burst_mode="adaptive", burst_t_max=50,
                     patience_min_gain=1000.0)  # fmt: skip
    system, ctx = _ctx(checkpoint, config, baseline_canary=100.0)
    before = {n: system._weight_matrix(h).clone() for n, h in system.handles.items()}
    steps = {"n": 0}
    for cls in (torch.optim.SGD, torch.optim.Adam):  # whichever the config selects
        monkeypatch.setattr(cls, "step", _counting(cls.step, steps))
    fields = build_policy(config).on_wave(ctx)

    assert not fields["burst_skipped"]
    # plateau rule with kappa=1000, window 1, m=2: stops at the 2nd canary eval = batch 4
    assert fields["stop_reason"] == "plateau"
    assert fields["burst_batches"] == fields["optimizer_steps"] == steps["n"] == 4
    assert [t[0] for t in fields["canary_trace"]] == [2, 4]
    trained = unchanged = 0
    for name, handle in system.handles.items():
        mon = system.monitors[name]
        now = system._weight_matrix(handle)
        spare = mon.remapped_mask & ~mon.effective_stuck
        if spare.any():
            trained += int((now[spare] != before[name][spare]).sum())
        stuck = mon.effective_stuck
        assert torch.equal(now[stuck], before[name][stuck])  # unremapped stuck cells stay frozen
        unchanged += int(stuck.sum())
    assert trained > 0, "living spares must be trainable"
    assert unchanged > 0


def test_adaptive_burst_stops_at_t_max(checkpoint):
    config = _config(checkpoint, remap_order="magnitude", burst_mode="adaptive", burst_t_max=5,
                     patience_window=50)  # fmt: skip
    _, ctx = _ctx(checkpoint, config, baseline_canary=100.0)
    fields = build_policy(config).on_wave(ctx)
    assert fields["stop_reason"] == "t_max"
    assert fields["burst_batches"] == fields["optimizer_steps"] == 5


def test_priors_overlay_overrides_config_and_is_recorded(checkpoint, tmp_path):
    priors = tmp_path / "priors.json"
    priors.write_text(json.dumps({
        "eta": 5e-3, "delta": 2.5, "kappa": 0.3, "k": 4, "m": 10, "t_max": 77, "pacing": "greedy",
    }))  # fmt: skip
    config = _config(checkpoint, priors_path=str(priors))
    params, from_priors = effective_params(config)
    assert params["burst_lr"] == 5e-3 and params["canary_gate_delta"] == 2.5
    assert params["patience_window"] == 4 and params["patience_eval_every"] == 10
    # pacing is structural (it is what D-nopace toggles), never overlaid from priors
    assert params["burst_t_max"] == 77 and params["remap_pacing"] == "paced"
    assert set(from_priors) == {
        "burst_lr", "canary_gate_delta", "patience_min_gain", "patience_window",
        "patience_eval_every", "burst_t_max",
    }  # fmt: skip
    # config stays authoritative without a priors file
    plain = _config(checkpoint, burst_lr=2e-3)
    assert effective_params(plain) == (
        {
            "burst_lr": 2e-3, "canary_gate_delta": 1.0, "patience_min_gain": 0.5,
            "patience_window": 1, "patience_eval_every": 2, "burst_t_max": 1000,
            "remap_pacing": "paced",
        },
        [],
    )  # fmt: skip

    _, ctx = _ctx(
        checkpoint, _config(checkpoint, remap_order="magnitude", priors_path=str(priors)), 100.0
    )
    fields = build_policy(ctx.config).on_wave(ctx)
    # priors said greedy, config says paced: config wins, so wave 0 of 3 is paced
    assert fields["rho"] == 1 / 3 and fields["budget_by_layer"] is not None
    assert fields["effective_params"]["burst_lr"] == 5e-3
    assert "burst_lr" in fields["params_from_priors"]


def test_missing_or_bad_priors_fail_loudly(checkpoint, tmp_path):
    config = _config(checkpoint, priors_path=str(tmp_path / "nope.json"))
    with pytest.raises(FileNotFoundError, match="calibration"):
        run_deployment(config, tmp_path)
    with pytest.raises(FileNotFoundError):
        build_policy(config)
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"eta": 1e-3}))
    with pytest.raises(ValueError, match="lacks required keys"):
        build_policy(_config(checkpoint, priors_path=str(bad)))


def _load_script():
    spec = importlib.util.spec_from_file_location(
        "calibrate_retrain_controller", ROOT / "scripts" / "calibrate_retrain_controller.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def sweep(checkpoint, tmp_path_factory):
    """3 fresh-rebuild runs (each ~30s on CPU: CrossSim patching + calibration): both
    fractions, two etas at 0.1 (the reference fraction needs >= 2 etas to be chosen)."""
    out = tmp_path_factory.mktemp("sweep")
    runs = [(0.05, 1e-3), (0.1, 1e-3), (0.1, 1e-2)]
    data = run_sweep(_config(checkpoint, name="sweep"), out, runs, batches=4, eval_every=2)
    return out, data


def test_calibration_sweep_output_and_same_fault_map_across_etas(sweep):
    out, data = sweep
    on_disk = json.loads((out / "canary_curves.json").read_text())
    assert on_disk["curves"].keys() == data["curves"].keys() == {"0.05", "0.1"}
    assert set(on_disk["curves"]["0.1"]) == {"0.001", "0.01"}
    assert set(on_disk["curves"]["0.05"]) == {"0.001"}
    for per_eta in on_disk["curves"].values():
        for curve in per_eta.values():
            assert [b for b, _ in curve] == [0, 2, 4]
    # same fault map (fresh rebuild + same dedicated generator) => identical batch-0 canary
    # across etas, and the clean canary is the same in every run
    post_wave = {curve[0][1] for curve in on_disk["curves"]["0.1"].values()}
    assert len(post_wave) == 1
    assert on_disk["metadata"]["clean_acc"] >= 0 and on_disk["metadata"]["remap"] is False


@pytest.mark.skipif(
    not (ROOT / "scripts" / "calibrate_retrain_controller.py").exists(),
    reason="scripts/calibrate_retrain_controller.py is not part of this release",
)
def test_sweep_feeds_calibration_script_and_priors_drive_deployment(checkpoint, sweep, tmp_path):
    out, _ = sweep
    priors_path = tmp_path / "priors.json"
    script = _load_script()
    rc = script.main(
        ["--from-calibration-sweep", str(out), "--out", str(priors_path), "--lam", "0.002"]
    )
    assert rc == 0
    priors = json.loads(priors_path.read_text())
    assert {"eta", "delta", "kappa", "k", "m", "t_max", "pacing"} <= set(priors)
    assert priors["eta"] in (1e-3, 1e-2)
    assert priors["reference_fraction"] == 0.1
    src = priors["model_points"][repr(priors["eta"])]["source"]
    assert src["0.05"] in ("measured", "borrowed_from_eta=0.001") and src["0.1"] == "measured"
    assert "extrapolated" in src.values()  # the simulated lifetime reaches 0.3 > 0.1

    # the produced file drives a real D lifetime
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _, _, ev = _run(checkpoint, run_dir, name="d_priors", priors_path=str(priors_path))
    for e in ev["events"]:
        assert e["effective_params"]["burst_lr"] == priors["eta"]
        assert len(e["params_from_priors"]) == 6  # pacing is structural, never from priors


@pytest.mark.skipif(
    not (ROOT / "scripts" / "calibrate_retrain_controller.py").exists(),
    reason="scripts/calibrate_retrain_controller.py is not part of this release",
)
def test_legacy_single_fraction_format_still_consumed(tmp_path):
    script = _load_script()
    legacy = {
        "metadata": {"n_canary": 1000, "wave_fraction": 0.1, "clean_acc": 90.0},
        "curves": {
            "0.001": [[b, 60.0 + 20.0 * (1 - math.exp(-b / 200.0))] for b in range(0, 400, 25)],
            "0.0001": [[b, 60.0 + 10.0 * (1 - math.exp(-b / 400.0))] for b in range(0, 400, 25)],
        },
    }
    d = tmp_path / "legacy"
    d.mkdir()
    (d / "canary_curves.json").write_text(json.dumps(legacy))
    models, _, ref_f, prov = script.build_recovery_models(legacy, u_end=0.3)
    assert ref_f == 0.1 and set(models) == {1e-3, 1e-4}
    m = models[1e-3]
    assert m.stuck_fractions == [0.0, 0.1, 0.3]
    assert m.a_post[1] == pytest.approx(60.0) and m.a_inf[1] == pytest.approx(80.0, abs=1.0)
    assert prov[repr(1e-3)]["0.3"] == "extrapolated"
    assert script.main(["--from-calibration-sweep", str(d), "--out", str(d / "p.json")]) == 0
