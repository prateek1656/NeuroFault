"""The validator's whole job is to catch bad data - so these tests construct
deliberately-corrupted synthetic metrics/config dicts and confirm each one
actually flips the verdict (not just that a "clean" case passes), plus a
clean case in both directions, plus a check against real, already-generated
results/*/ directories on disk (not just synthetic data).
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from neurofault.validation.results_validator import Level, overall_verdict, validate


def _clean_metrics() -> dict:
    return {
        "epoch": [0, 0, 1, 1, 2],
        "batch": [0, 3, 6, 9, 12],
        "loss": [2.3, 2.1, 1.9, 1.7, 1.5],
        "accuracy": [10.0, 15.0, 20.0, 25.0, 30.0],
        "health": [95.0, 94.0, 93.0, 92.0, 91.0],
        "faults": [0, 10, 10, 20, 20],
        "soft_mitigations": [0, 0, 0, 0, 0],
        "remappings": [0, 0, 0, 0, 0],
        "layer_resets": [0, 0, 0, 0, 0],
        "detection_cost": [0, 5, 10, 15, 20],
    }


def _clean_config(name: str = "clean_test") -> dict:
    return {
        "name": name,
        "mitigation": {
            "enable_soft_mitigation": False,
            "enable_remapping": False,
            "enable_layer_reset": False,
        },
        "fault": {"density": 0.1, "failure_model": "density"},
        "retraining": {"enable_fault_aware": False},
    }


def _find(results, name: str):
    for r in results:
        if r.name == name:
            return r
    raise AssertionError(f"no check named {name!r} in results: {[r.name for r in results]}")


def test_clean_metrics_produce_overall_pass():
    results = validate(_clean_metrics(), _clean_config(), results_name="clean_test")
    assert overall_verdict(results) == Level.PASS
    assert all(r.level == Level.PASS for r in results)


class TestStructuralChecksCatchCorruption:
    def test_missing_key_fails(self):
        metrics = _clean_metrics()
        del metrics["health"]
        results = validate(metrics, _clean_config())
        assert overall_verdict(results) == Level.FAIL
        assert _find(results, "structural.missing_keys").level == Level.FAIL

    def test_mismatched_list_lengths_fails(self):
        metrics = _clean_metrics()
        metrics["loss"] = metrics["loss"][:-1]  # one shorter than the rest
        results = validate(metrics, _clean_config())
        assert overall_verdict(results) == Level.FAIL
        assert _find(results, "structural.mismatched_lengths").level == Level.FAIL

    def test_negative_epoch_or_batch_fails(self):
        metrics = _clean_metrics()
        metrics["batch"][0] = -1
        results = validate(metrics, _clean_config())
        assert _find(results, "structural.negative_counters").level == Level.FAIL

    def test_non_monotonic_batch_fails(self):
        metrics = _clean_metrics()
        metrics["batch"] = [0, 3, 2, 9, 12]  # decreases at index 2
        results = validate(metrics, _clean_config())
        assert _find(results, "structural.batch_not_monotonic").level == Level.FAIL

    def test_name_mismatch_fails(self):
        results = validate(_clean_metrics(), _clean_config(name="run_a"), results_name="run_b")
        assert _find(results, "structural.name_mismatch").level == Level.FAIL


class TestNumericSanityChecksCatchGibberish:
    def test_nan_in_loss_fails(self):
        metrics = _clean_metrics()
        metrics["loss"][2] = float("nan")
        results = validate(metrics, _clean_config())
        assert overall_verdict(results) == Level.FAIL
        assert _find(results, "numeric.loss_finite").level == Level.FAIL

    def test_inf_in_accuracy_fails(self):
        metrics = _clean_metrics()
        metrics["accuracy"][1] = float("inf")
        results = validate(metrics, _clean_config())
        assert _find(results, "numeric.accuracy_finite").level == Level.FAIL

    def test_accuracy_above_100_fails(self):
        metrics = _clean_metrics()
        metrics["accuracy"][-1] = 150.0
        results = validate(metrics, _clean_config())
        assert _find(results, "numeric.accuracy_bounds").level == Level.FAIL

    def test_health_below_zero_fails(self):
        metrics = _clean_metrics()
        metrics["health"][0] = -5.0
        results = validate(metrics, _clean_config())
        assert _find(results, "numeric.health_bounds").level == Level.FAIL

    def test_negative_loss_fails(self):
        metrics = _clean_metrics()
        metrics["loss"][0] = -1.0
        results = validate(metrics, _clean_config())
        assert _find(results, "numeric.loss_non_negative").level == Level.FAIL

    def test_decreasing_cumulative_counter_fails(self):
        metrics = _clean_metrics()
        metrics["faults"] = [0, 10, 10, 5, 20]  # drops from 10 to 5
        results = validate(metrics, _clean_config())
        assert _find(results, "numeric.faults_monotonic").level == Level.FAIL

    def test_negative_counter_fails(self):
        metrics = _clean_metrics()
        metrics["soft_mitigations"][0] = -1
        results = validate(metrics, _clean_config())
        assert _find(results, "numeric.soft_mitigations_non_negative").level == Level.FAIL

    def test_grad_norm_absent_emits_no_check(self):
        """grad_norm is optional (added after 67 real results already existed
        without it) - a metrics.json without it must still validate exactly
        as it did before this field existed."""
        results = validate(_clean_metrics(), _clean_config())
        assert all(r.name != "numeric.grad_norm_finite" for r in results)
        assert overall_verdict(results) == Level.PASS

    def test_grad_norm_present_and_finite_passes(self):
        metrics = _clean_metrics()
        metrics["grad_norm"] = [1.2, 1.1, 0.9, 0.8, 0.7]
        results = validate(metrics, _clean_config())
        assert _find(results, "numeric.grad_norm_finite").level == Level.PASS

    def test_grad_norm_nan_fails(self):
        metrics = _clean_metrics()
        metrics["grad_norm"] = [1.2, 1.1, float("nan"), 0.8, 0.7]
        results = validate(metrics, _clean_config())
        assert _find(results, "numeric.grad_norm_finite").level == Level.FAIL


class TestConfigConsistencyChecksCatchDispatchViolations:
    def test_soft_mitigation_disabled_but_metric_nonzero_fails(self):
        """The artifact-level version of the exact bug fixed in
        tests/invariants/: soft mitigation firing despite being disabled."""
        metrics = _clean_metrics()
        metrics["soft_mitigations"] = [0, 0, 1, 1, 1]
        config = _clean_config()  # enable_soft_mitigation=False
        results = validate(metrics, config)
        assert overall_verdict(results) == Level.FAIL
        assert _find(results, "consistency.enable_soft_mitigation").level == Level.FAIL

    def test_remapping_disabled_but_metric_nonzero_fails(self):
        metrics = _clean_metrics()
        metrics["remappings"] = [0, 2, 2, 2, 2]
        results = validate(metrics, _clean_config())
        assert _find(results, "consistency.enable_remapping").level == Level.FAIL

    def test_zero_density_baseline_with_faults_fails(self):
        metrics = _clean_metrics()
        config = _clean_config()
        config["fault"]["density"] = 0
        results = validate(metrics, config)  # faults is non-zero in _clean_metrics()
        assert _find(results, "consistency.zero_density_baseline").level == Level.FAIL

    def test_fault_aware_retraining_with_multiple_injections_fails(self):
        metrics = _clean_metrics()
        metrics["faults"] = [0, 10, 20, 20, 20]  # grew twice - periodic re-injection happened
        config = _clean_config()
        config["retraining"]["enable_fault_aware"] = True
        results = validate(metrics, config)
        assert (
            _find(results, "consistency.fault_aware_retraining_single_injection").level
            == Level.FAIL
        )

    def test_fault_aware_retraining_with_single_injection_passes(self):
        metrics = _clean_metrics()
        metrics["faults"] = [0, 20, 20, 20, 20]  # grew once, then fixed
        config = _clean_config()
        config["retraining"]["enable_fault_aware"] = True
        results = validate(metrics, config)
        assert (
            _find(results, "consistency.fault_aware_retraining_single_injection").level
            == Level.PASS
        )


class TestQualitativeChecksWarnNotFail:
    def test_accuracy_near_random_chance_warns(self):
        metrics = _clean_metrics()
        metrics["accuracy"] = [10.0, 11.0, 9.0, 10.0, 11.0]
        results = validate(metrics, _clean_config())
        assert overall_verdict(results) == Level.WARN  # not FAIL
        assert _find(results, "qualitative.final_accuracy").level == Level.WARN

    def test_loss_not_decreasing_warns(self):
        metrics = _clean_metrics()
        metrics["loss"] = [1.5, 1.6, 1.7, 1.8, 1.9]
        results = validate(metrics, _clean_config())
        assert _find(results, "qualitative.loss_trend").level == Level.WARN

    def test_flat_accuracy_warns(self):
        metrics = _clean_metrics()
        metrics["accuracy"] = [20.0] * 5
        results = validate(metrics, _clean_config())
        assert _find(results, "qualitative.accuracy_flat").level == Level.WARN


class TestOutlierChecksWarnNotFail:
    def test_large_accuracy_jump_warns(self):
        metrics = _clean_metrics()
        metrics["accuracy"] = [10.0, 15.0, 90.0, 25.0, 30.0]  # +75 then -65
        results = validate(metrics, _clean_config())
        assert overall_verdict(results) == Level.WARN
        assert _find(results, "outlier.accuracy_jump").level == Level.WARN

    def test_large_loss_jump_warns(self):
        metrics = _clean_metrics()
        metrics["loss"] = [2.3, 2.1, 25.0, 1.7, 1.5]  # 12x jump then back down
        results = validate(metrics, _clean_config())
        assert _find(results, "outlier.loss_jump").level == Level.WARN


class TestAgainstRealGeneratedResults:
    """Not just synthetic data - the validator must also behave sensibly on
    the real, already-generated experiment output."""

    RESULTS_ROOT = Path(__file__).resolve().parents[2] / "results"

    @pytest.mark.parametrize(
        "name", ["no_mitigation", "baseline_no_fault", "fault_aware_retraining"]
    )
    def test_real_results_directory_has_no_fail_level_checks(self, name):
        results_dir = self.RESULTS_ROOT / name
        if not (results_dir / "metrics.json").exists():
            pytest.skip(f"results/{name} not present in this checkout")

        metrics = json.loads((results_dir / "metrics.json").read_text())
        config = json.loads((results_dir / "config.json").read_text())

        results = validate(metrics, config, results_name=name)

        failures = [r for r in results if r.level == Level.FAIL]
        assert failures == [], f"real results/{name} unexpectedly failed validation: {failures}"


def test_validate_does_not_mutate_its_inputs():
    """A validator that mutates the caller's dicts would be a subtle, nasty
    bug in the thing meant to catch subtle, nasty bugs."""
    metrics = _clean_metrics()
    config = _clean_config()
    metrics_copy = copy.deepcopy(metrics)
    config_copy = copy.deepcopy(config)

    validate(metrics, config, results_name="clean_test")

    assert metrics == metrics_copy
    assert config == config_copy
