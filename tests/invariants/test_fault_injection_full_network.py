"""Closes a real coverage gap found while planning the fault-injection campaign on the
ResNet-9 architecture: every existing test exercises fault injection/mitigation against at most one small crossbar
(`patch_layer_types=["Linear"]`). Nothing in the repo had ever run the full 9-crossbar
network (sizes from ~1.7K to 2.36M cells, patched with `["Linear", "Conv2d"]`) through a
real fault-injection + mitigation-dispatch cycle before this file, on either the new
ADC/DAC-calibrated backend or the old one.

Also regression-guards a real finding from that same planning pass: `soft_mitigation`
only ever acts on `monitor.drift_mask` (`mitigation/dispatch.py`), which is set exclusively
by `failure_model="drift"` - never by the default `"density"` (stuck-at) model every
existing experiment config uses. Verified directly (same seed, `enable_soft_mitigation` off
vs on, `failure_model="density"`): bit-identical loss/accuracy trajectories, soft-mitigation
count stays zero in both. `test_soft_mitigation_is_a_noop_under_density_faults` below pins
that down as expected, documented behavior rather than a silent gap - and
`test_soft_mitigation_fires_under_drift_faults` confirms the mechanism does work once given
the failure mode it targets.

Marked `slow` for the same reason as `test_end_to_end_canary.py` - a real (if tiny) training
run through 9 real crossbars, not a fast unit test.
"""

from __future__ import annotations

import pytest

from neurofault.config import ExperimentConfig, FaultConfig

pytest.importorskip("simulator", reason="CrossSim not installed (uv sync --extra crosssim)")

from experiments.run import run_experiment

pytestmark = pytest.mark.slow


def _full_network_config(failure_model: str = "density") -> ExperimentConfig:
    config = ExperimentConfig(
        name="fault_full_network_probe",
        simulator="crosssim",
        seed=42,
        epochs=3,
        num_batches=16,
        train_samples=256,
        test_samples=64,
        batch_size=16,
        eval_interval=1,
        learning_rate=0.01,
    )
    config.crossbar.patch_layer_types = ["Linear", "Conv2d"]
    config.fault.density = 0.1
    config.fault.distribution = "random"
    config.fault.injection_interval = 4
    config.fault.failure_model = failure_model
    return config


def _no_mitigation_config() -> ExperimentConfig:
    config = _full_network_config()
    config.mitigation.enable_soft_mitigation = False
    config.mitigation.enable_remapping = False
    config.mitigation.enable_layer_reset = False
    config.mitigation.enable_predictive = False
    return config


def test_faults_are_injected_into_every_patched_crossbar_size():
    """Sanity check for the actual scenario this file exists to cover: 9 crossbars of
    very different sizes (~1.7K to 2.36M cells), all faulted in the same run."""
    metrics = run_experiment(_no_mitigation_config())
    assert metrics["faults"][0] > 0, "no faults were injected at all"
    assert metrics["faults"][-1] > metrics["faults"][0], (
        "fault count never grew - injection_interval isn't firing repeatedly"
    )


def test_no_mitigation_arm_never_dispatches_any_strategy():
    metrics = run_experiment(_no_mitigation_config())
    assert all(count == 0 for count in metrics["soft_mitigations"])
    assert all(count == 0 for count in metrics["remappings"])
    assert all(count == 0 for count in metrics["layer_resets"])


def test_soft_mitigation_is_a_noop_under_density_faults():
    """Pins down the real finding: enabling soft_mitigation against the default
    ("density"/stuck-at) fault model changes nothing, because it only ever acts on
    drift_mask, which this fault model never sets. Documented expected behavior, not a
    bug - see this file's module docstring."""
    config = _full_network_config()
    config.mitigation.enable_soft_mitigation = True
    metrics = run_experiment(config)
    assert all(count == 0 for count in metrics["soft_mitigations"])


def test_soft_mitigation_fires_under_drift_faults():
    """The mechanism does work once it has the failure mode it actually targets."""
    config = _full_network_config("drift")
    config.mitigation.enable_soft_mitigation = True
    metrics = run_experiment(config)
    assert metrics["soft_mitigations"][-1] > 0


def test_remapping_arm_dispatches_real_remaps():
    config = _full_network_config()
    config.mitigation.enable_remapping = True
    # remap only fires while avg_health > health_threshold (75, dispatch.py) - this
    # config's fast injection_interval=4 (deliberately aggressive, to fit several
    # injection cycles into a tiny test) decays health past that band before the
    # default health_check_interval=10 ever runs a check with faults present
    # (verified directly: health_check_interval=10 -> 0 remaps, =1 -> 656870).
    # Checking every batch here is a test-calibration fix, not a claim about the
    # real full-scale campaign, where injection_interval will be ~1000s of batches
    # and this timing gap won't arise.
    config.mitigation.health_check_interval = 1
    metrics = run_experiment(config)
    assert metrics["remappings"][-1] > 0


def test_layer_reset_arm_dispatches_real_resets():
    config = _full_network_config()
    config.mitigation.enable_layer_reset = True
    metrics = run_experiment(config)
    assert metrics["layer_resets"][-1] > 0


def test_combined_arm_dispatches_remap_and_layer_reset_but_not_soft():
    config = _full_network_config()
    config.mitigation.enable_soft_mitigation = True
    config.mitigation.enable_remapping = True
    config.mitigation.enable_layer_reset = True
    # See test_remapping_arm_dispatches_real_remaps's comment - same timing fix,
    # needed here for the same reason (remap's >75 health window closes fast).
    config.mitigation.health_check_interval = 1
    metrics = run_experiment(config)
    assert all(count == 0 for count in metrics["soft_mitigations"])
    assert metrics["remappings"][-1] > 0
    assert metrics["layer_resets"][-1] > 0


def test_mixed_regime_fires_soft_mitigation_without_losing_stuck_at_faults():
    """Regression test for the real fix this file's module docstring flags as
    needed: secondary_fault (config.py) lets density (stuck-at) and drift fire
    simultaneously, so soft_mitigation finally gets real work (unlike
    test_soft_mitigation_is_a_noop_under_density_faults above) in the SAME run
    that's also injecting stuck-at faults - not a separate drift-only run."""
    config = _full_network_config()  # primary: density/stuck-at, as every arm uses
    config.secondary_fault = FaultConfig(
        failure_model="drift",
        density=0.1,
        distribution="random",
        injection_interval=config.fault.injection_interval,
    )
    config.mitigation.enable_soft_mitigation = True
    # Real failure hit running this test for real (not assumed): soft_mitigation is gated by
    # the identical `avg_health > health_threshold` dispatch.py band as remap (see
    # test_remapping_arm_dispatches_real_remaps's comment above) - with this tiny test's
    # aggressive injection_interval=4, stuck-at faults alone crash health past that band
    # before a default health_check_interval=10 check ever runs with faults present. Same
    # test-calibration fix as the remap/combined tests above, not a secondary_fault bug.
    config.mitigation.health_check_interval = 1
    metrics = run_experiment(config)
    assert metrics["faults"][-1] > 0, "stuck-at faults from the primary channel vanished"
    assert metrics["soft_mitigations"][-1] > 0, (
        "soft_mitigation still didn't fire - secondary_fault's drift isn't reaching drift_mask"
    )
