"""Closed loop X.1: controller D on an ESTIMATED fault map, through the real deployment loop
(tiny CPU config, real CrossSim layers). The oracle path must be untouched."""

from __future__ import annotations

import pytest
import torch

pytest.importorskip("simulator", reason="CrossSim not installed (uv sync --extra crosssim)")

import experiments.run as run_mod
from neurofault.spectral.closed_loop import LayerEstimate, ProbeEstimator

from .test_co_mitigation import _config, checkpoint  # noqa: F401  (fixture)


def _run(checkpoint, tmp_path, monkeypatch, estimator):  # noqa: F811
    load = run_mod._load_deployment_system

    def patched(config, device):
        system, *rest = load(config, device)
        system.fault_map_source = "estimated"
        system.fault_estimator = estimator
        return (system, *rest)

    monkeypatch.setattr(run_mod, "_load_deployment_system", patched)
    config = _config(checkpoint)
    return run_mod.run_deployment(config, tmp_path)["events"]["events"]


def test_oracle_path_has_no_localize_fields(checkpoint, tmp_path):  # noqa: F811
    events = run_mod.run_deployment(_config(checkpoint), tmp_path)["events"]["events"]
    assert all("localize" not in e for e in events)


def test_estimated_map_drives_d_and_matches_oracle_remaps(checkpoint, tmp_path, monkeypatch):  # noqa: F811
    (tmp_path / "o").mkdir()
    (tmp_path / "e").mkdir()
    oracle = run_mod.run_deployment(_config(checkpoint), tmp_path / "o")["events"]["events"]
    est = _run(checkpoint, tmp_path / "e", monkeypatch,
               ProbeEstimator(kind="readout", sigma_frac=0.005, seed=1))  # fmt: skip
    for ee in est:
        loc = ee["localize"]
        assert loc["reads"] > 0 and loc["tiles"] > 0
        assert loc["found_new_visible"] >= 0.95 * loc["truth_new_visible"]
        assert loc["fn_visible"] == 0  # misses are only cells stuck near their own value
        # false alarms only at the designed rate (fa = 0.1 per tile per call, cumulative)
        assert loc["fp"] <= 0.1 * loc["tiles"] * (loc["call"] + 1) + 10
    # a near-perfect map spends spares like the oracle does (wave 0 is identical state)
    assert len(est) == len(oracle)
    assert abs(est[0]["remapped_this_wave"] - oracle[0]["remapped_this_wave"]) <= 2


def test_decisions_read_the_belief_not_the_truth(checkpoint, tmp_path, monkeypatch):  # noqa: F811
    def blind(name, read_matrix, w_ref, call):
        z = torch.zeros(w_ref.shape, dtype=torch.bool)
        return LayerEstimate(z, torch.zeros(w_ref.shape, dtype=torch.float64))

    events = _run(checkpoint, tmp_path, monkeypatch, blind)
    assert all(e["remapped_this_wave"] == 0 for e in events)
    assert all(e["unremapped_believed_stuck"] == 0 for e in events)
    assert events[-1]["unremapped_effective_stuck"] > 0  # truth still has faults


def test_estimated_remap_of_a_false_positive_in_a_fault_free_layer(checkpoint):  # noqa: F811
    """Regression (box smoke 2026-10-12): a believed cell in a layer with no injected fault yet
    (no value-at-fault snapshot) must remap without error and keep its digital weight."""
    from neurofault.data import load_datasets
    from neurofault.models.cnn import build_model

    config = _config(checkpoint)
    trainloader, _ = load_datasets(config)
    base = build_model(config)
    base.load_state_dict(torch.load(checkpoint, weights_only=False)["model_state_dict"])
    system = run_mod._build_system(config, base, trainloader, torch.device("cpu"))
    system.fisher = None
    system.fault_map_source = "estimated"
    name = next(iter(system.handles))
    monitor = system.monitors[name]
    monitor.believed_stuck[0, 0] = True
    w = system._weight_matrix(system.handles[name])
    before = float(w[0, 0])
    assert system.remap_now(order="random")[name] == 1
    assert float(w[0, 0]) == before
    assert not monitor.decision_stuck[0, 0]
