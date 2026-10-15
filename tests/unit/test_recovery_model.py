"""Recovery fit, patience stopper, offline simulator and calibration script."""

from __future__ import annotations

import importlib.util
import json
from itertools import pairwise
from pathlib import Path

import numpy as np
import pytest

from neurofault.retraining.recovery_model import (
    PatienceStopper,
    PolicyParams,
    RecoveryModel,
    Wave,
    fit_exponential_recovery,
    grid_search,
    pareto_front,
    simulate_lifetime,
)

ROOT = Path(__file__).resolve().parents[2]


def _load_script():
    spec = importlib.util.spec_from_file_location(
        "calibrate_retrain_controller", ROOT / "scripts" / "calibrate_retrain_controller.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _model(tau=800.0, eta=1e-3):
    return RecoveryModel(
        eta=eta,
        tau=tau,
        stuck_fractions=[0.0, 0.1, 0.2, 0.3],
        a_inf=[90.0, 80.0, 70.0, 60.0],
        a_post=[90.0, 40.0, 30.0, 20.0],
    )


WAVES = [Wave(1000, 0.1), Wave(5000, 0.1), Wave(9000, 0.1)]


class TestFit:
    def test_recovers_known_params(self):
        rng = np.random.default_rng(0)
        t = np.arange(0, 4000, 100.0)
        y = 70.0 - (70.0 - 15.0) * np.exp(-t / 900.0) + rng.normal(0, 0.5, t.size)
        fit = fit_exponential_recovery(t, y)
        assert fit.converged
        assert fit.tau == pytest.approx(900.0, rel=0.1)
        assert fit.a_inf == pytest.approx(70.0, abs=1.5)
        assert fit.a0 == pytest.approx(15.0, abs=2.0)

    @pytest.mark.parametrize("t,y", [([], []), ([0, 1], [1, 2]), ([0, 1, 2, 3], [np.nan] * 4)])
    def test_short_series_falls_back(self, t, y):
        fit = fit_exponential_recovery(t, y)
        assert not fit.converged


class TestStopper:
    def test_stops_on_plateau(self):
        s = PatienceStopper(25, 3, 0.1, 10_000)
        stopped = [s.update(80.0) for _ in range(6)]
        assert stopped[-1] and s.reason == "plateau"

    def test_stops_on_t_max(self):
        s = PatienceStopper(25, 3, -1e9, 100)
        for i in range(4):
            s.update(50.0 + i * 5)
        assert s.reason == "t_max" and s.batches == 100

    def test_no_early_stop_on_rising_noisy_curve(self):
        rng = np.random.default_rng(1)
        false_stops = 0
        for _ in range(100):
            s = PatienceStopper(25, 6, 0.1, 10_000)
            for i in range(24):  # +0.5 pt/eval, canary SE ~1 pt
                if s.update(50 + 0.5 * i + rng.normal(0, 1.0)):
                    false_stops += 1
                    break
        assert false_stops <= 5


class TestSimulator:
    def _run(self, capacity=0.0, pacing="greedy", t_max=1000, kappa=0.1, noise=False):
        p = PolicyParams(1e-3, -1.0, kappa, 3, 25, t_max, pacing)
        return simulate_lifetime(
            p, {1e-3: _model()}, WAVES, capacity, horizon=13000, noise=noise, n_rep=4
        )

    def test_more_capacity_never_lowers_lifetime_acc(self):
        means = [self._run(capacity=c).lifetime_mean_acc for c in (0.0, 0.05, 0.1, 0.2, 0.3)]
        assert all(b >= a - 1e-9 for a, b in pairwise(means))

    def test_longer_bursts_never_lower_final_acc(self):
        finals = [self._run(t_max=t, kappa=-1e9).final_acc for t in (100, 300, 1000, 3000)]
        assert all(b >= a - 1e-9 for a, b in pairwise(finals))

    def test_gate_skips_burst(self):
        p = PolicyParams(1e-3, 1000.0, 0.1, 3, 25, 1000)
        r = simulate_lifetime(p, {1e-3: _model()}, WAVES, 0.0, noise=False)
        assert r.burst_batches == 0

    def test_noisy_run_is_seeded(self):
        a, b = self._run(noise=True), self._run(noise=True)
        assert a.lifetime_mean_acc == b.lifetime_mean_acc


class TestGridSearch:
    def test_pareto_is_non_dominated_and_contains_chosen(self):
        res = grid_search(
            {1e-3: _model()},
            WAVES,
            0.1,
            lam=0.002,
            kappas=(0.05, 0.25),
            ks=(3,),
            t_maxs=(200, 1000),
            horizon=13000,
            n_rep=3,
        )
        assert len(res.pareto) >= 1
        for p in res.pareto:
            assert not any(
                q.result.lifetime_mean_acc >= p.result.lifetime_mean_acc
                and q.result.burst_batches <= p.result.burst_batches
                and (
                    q.result.lifetime_mean_acc > p.result.lifetime_mean_acc
                    or q.result.burst_batches < p.result.burst_batches
                )
                for q in res.points
            )
        assert res.chosen in res.pareto
        assert pareto_front(res.pareto) == res.pareto


@pytest.mark.skipif(
    not (ROOT / "scripts" / "calibrate_retrain_controller.py").exists(),
    reason="scripts/calibrate_retrain_controller.py is not part of this release",
)
class TestScript:
    def test_past_results_reproduces_no_mitigation_fits(self):
        results = ROOT / "results"
        run = results / "resnet9_fault_no_mitigation_seed42"
        if not (run / "metrics.json").exists():
            pytest.skip("real results not present")
        mod = _load_script()
        metrics = json.loads((run / "metrics.json").read_text())
        rows = mod.fit_wave_segments(metrics)
        assert [r["batch"] for r in rows] == [5200, 10400, 15600]
        for r, tau, a_inf in zip(rows, (1308, 1086, 593), (67.8, 57.5, 39.2)):
            assert r["fit"].tau == pytest.approx(tau, rel=0.1)
            assert r["fit"].a_inf == pytest.approx(a_inf, rel=0.1)

    def test_calibration_sweep_mode_on_synthetic_fixture(self, tmp_path):
        rng = np.random.default_rng(0)
        t = np.arange(0, 1000, 25)
        curves = {}
        for eta, tau, a_inf in ((1e-4, 1500, 70), (1e-3, 600, 78), (5e-3, 300, 74)):
            y = a_inf - (a_inf - 40) * np.exp(-t / tau) + rng.normal(0, 0.5, t.size)
            curves[str(eta)] = [[int(b), float(a)] for b, a in zip(t, y)]
        (tmp_path / "canary_curves.json").write_text(
            json.dumps(
                {
                    "metadata": {
                        "n_canary": 1000,
                        "wave_fraction": 0.1,
                        "clean_acc": 89.8,
                        "seed": 42,
                    },
                    "curves": curves,
                }
            )
        )
        out = tmp_path / "priors.json"
        mod = _load_script()
        assert mod.main(["--from-calibration-sweep", str(tmp_path), "--out", str(out)]) == 0
        priors = json.loads(out.read_text())
        assert priors["eta"] in (1e-4, 1e-3, 5e-3)
        assert {"delta", "kappa", "k", "m", "t_max", "rho_schedule", "provenance"} <= set(priors)
        assert set(priors["fitted_per_eta"]) == set(curves)
