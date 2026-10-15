"""Systematic sweep, not spot checks: every config field must either (a)
reject an out-of-physical-range value clearly, or (b) be a deliberate,
documented "anything goes" field. Silence by omission is what let a typo'd
redundancy_scheme run an entire experiment under the wrong scheme,
indistinguishable from a correctly-configured run (see
test_known_bug_regressions.py) - this file is the systematic version of that
one finding, covering every field, not just the two already caught.
"""

from __future__ import annotations

import pytest

from neurofault.config import (
    CrossbarConfig,
    DeviceConfig,
    ExperimentConfig,
    FaultConfig,
    MitigationConfig,
    RetrainingConfig,
)


class TestDeviceConfigValidation:
    def test_rejects_non_positive_r_on(self):
        with pytest.raises(ValueError, match="r_on/r_off must be positive"):
            DeviceConfig(r_on=0.0, r_off=1000.0)

    def test_rejects_non_positive_r_off(self):
        with pytest.raises(ValueError, match="r_on/r_off must be positive"):
            DeviceConfig(r_on=100.0, r_off=-5.0)

    def test_rejects_r_on_greater_than_or_equal_to_r_off(self):
        with pytest.raises(ValueError, match="r_on must be less than r_off"):
            DeviceConfig(r_on=1000.0, r_off=1000.0)
        with pytest.raises(ValueError, match="r_on must be less than r_off"):
            DeviceConfig(r_on=2000.0, r_off=1000.0)

    def test_rejects_non_positive_time_series_resolution(self):
        with pytest.raises(ValueError, match="time_series_resolution must be positive"):
            DeviceConfig(time_series_resolution=0.0)

    def test_accepts_valid_values(self):
        DeviceConfig(r_on=1.0, r_off=1.0001, time_series_resolution=1e-12)  # must not raise


class TestCrossbarConfigValidation:
    @pytest.mark.parametrize("adc_resolution,dac_resolution", [(0, 8), (8, 0), (-1, 8)])
    def test_rejects_non_positive_adc_dac_resolution(self, adc_resolution, dac_resolution):
        with pytest.raises(ValueError, match="adc_resolution/dac_resolution must be >= 1"):
            CrossbarConfig(adc_resolution=adc_resolution, dac_resolution=dac_resolution)

    @pytest.mark.parametrize("value", [-0.1, 1.1])
    def test_rejects_out_of_range_redundancy_factor(self, value):
        with pytest.raises(ValueError, match="redundancy_factor must be in"):
            CrossbarConfig(redundancy_factor=value)

    @pytest.mark.parametrize("value", [-0.1, 1.1])
    def test_rejects_out_of_range_condemn_threshold(self, value):
        with pytest.raises(ValueError, match="redundancy_condemn_threshold must be in"):
            CrossbarConfig(redundancy_condemn_threshold=value)

    def test_rejects_devices_per_synapse_below_one(self):
        with pytest.raises(ValueError, match="devices_per_synapse must be >= 1"):
            CrossbarConfig(devices_per_synapse=0)

    @pytest.mark.parametrize("tile_shape", [(0, 10), (10, 0), (-1, 10)])
    def test_rejects_non_positive_tile_shape_dimensions(self, tile_shape):
        with pytest.raises(ValueError, match="tile_shape dimensions must be positive"):
            CrossbarConfig(tile_shape=tile_shape)

    def test_rejects_unknown_circuit_topology(self):
        with pytest.raises(ValueError, match="Unknown crossbar_config.circuit_topology"):
            CrossbarConfig(circuit_topology="bogus")

    def test_accepts_none_circuit_topology(self):
        CrossbarConfig(circuit_topology=None)  # must not raise - the documented default

    def test_rejects_unknown_redundancy_scheme(self):
        with pytest.raises(ValueError, match="Unknown crossbar_config.redundancy_scheme"):
            CrossbarConfig(redundancy_scheme="bogus")

    def test_accepts_valid_values(self):
        CrossbarConfig(
            adc_resolution=1,
            redundancy_factor=0.0,
            redundancy_condemn_threshold=1.0,
            devices_per_synapse=1,
            tile_shape=(1, 1),
            circuit_topology="1T1R",
            redundancy_scheme="row_col_granularity",
        )  # must not raise - boundary values are valid, not just interior ones


class TestFaultConfigValidation:
    def test_rejects_unknown_failure_model(self):
        with pytest.raises(ValueError, match="Unknown FaultConfig.failure_model"):
            FaultConfig(failure_model="bogus")

    def test_rejects_unknown_distribution(self):
        with pytest.raises(ValueError, match="Unknown FaultConfig.distribution"):
            FaultConfig(distribution="bogus")

    @pytest.mark.parametrize("density", [-0.1, 1.1])
    def test_rejects_out_of_range_density(self, density):
        with pytest.raises(ValueError, match="density must be in"):
            FaultConfig(density=density)

    def test_accepts_density_boundary_values(self):
        FaultConfig(density=0.0)
        FaultConfig(density=1.0)

    def test_rejects_non_positive_injection_interval(self):
        with pytest.raises(ValueError, match="injection_interval must be >= 1"):
            FaultConfig(injection_interval=0)

    def test_rejects_non_positive_weibull_shape(self):
        with pytest.raises(ValueError, match="weibull_shape must be positive"):
            FaultConfig(weibull_shape=0.0)

    def test_rejects_non_positive_weibull_scale_when_set(self):
        with pytest.raises(ValueError, match="weibull_scale must be positive"):
            FaultConfig(failure_model="weibull", weibull_scale=-1.0)

    def test_rejects_non_positive_exponential_rate_when_set(self):
        with pytest.raises(ValueError, match="exponential_rate must be positive"):
            FaultConfig(failure_model="exponential", exponential_rate=0.0)

    @pytest.mark.parametrize(
        "temperature_kelvin,reference_temperature_kelvin", [(0.0, 300.0), (300.0, -1.0)]
    )
    def test_rejects_non_positive_temperatures(
        self, temperature_kelvin, reference_temperature_kelvin
    ):
        with pytest.raises(ValueError, match="temperature_kelvin"):
            FaultConfig(
                temperature_kelvin=temperature_kelvin,
                reference_temperature_kelvin=reference_temperature_kelvin,
            )

    def test_still_requires_weibull_scale_and_exponential_rate_explicitly(self):
        """Pre-existing checks (added earlier) must still work
        alongside the new ones added here."""
        with pytest.raises(ValueError, match="requires weibull_scale"):
            FaultConfig(failure_model="weibull")
        with pytest.raises(ValueError, match="requires exponential_rate"):
            FaultConfig(failure_model="exponential")


class TestMitigationConfigValidation:
    @pytest.mark.parametrize("value", [-0.1, 100.1])
    def test_rejects_out_of_range_health_threshold(self, value):
        with pytest.raises(ValueError, match="health_threshold must be in"):
            MitigationConfig(health_threshold=value)

    def test_accepts_health_threshold_boundary_values(self):
        MitigationConfig(health_threshold=0.0)
        MitigationConfig(health_threshold=100.0)

    def test_rejects_non_positive_health_check_interval(self):
        with pytest.raises(ValueError, match="health_check_interval must be >= 1"):
            MitigationConfig(health_check_interval=0)

    def test_rejects_unknown_detection_method(self):
        with pytest.raises(ValueError, match="Unknown MitigationConfig.detection_method"):
            MitigationConfig(detection_method="bogus")

    def test_rejects_negative_checksum_z_threshold(self):
        with pytest.raises(ValueError, match="checksum_z_threshold must be >= 0"):
            MitigationConfig(checksum_z_threshold=-0.01)


class TestExperimentConfigValidation:
    def test_rejects_unknown_simulator(self):
        with pytest.raises(ValueError, match="Unknown ExperimentConfig.simulator"):
            ExperimentConfig(name="x", simulator="bogus")

    def test_rejects_unknown_dataset(self):
        with pytest.raises(ValueError, match="Unknown ExperimentConfig.dataset"):
            ExperimentConfig(name="x", dataset="bogus")

    @pytest.mark.parametrize(
        "field,value",
        [
            ("train_samples", 0),
            ("test_samples", 0),
            ("batch_size", 0),
            ("epochs", 0),
            ("num_batches", 0),
            ("eval_interval", 0),
        ],
    )
    def test_rejects_non_positive_count_fields(self, field, value):
        with pytest.raises(ValueError):
            ExperimentConfig(name="x", **{field: value})

    def test_rejects_non_positive_learning_rate(self):
        with pytest.raises(ValueError, match="learning_rate must be positive"):
            ExperimentConfig(name="x", learning_rate=0.0)

    def test_default_construction_does_not_raise(self):
        ExperimentConfig(name="x")  # every default value must itself be valid

    def test_accepts_valid_secondary_fault(self):
        ExperimentConfig(
            name="x",
            fault=FaultConfig(failure_model="density", injection_interval=5200),
            secondary_fault=FaultConfig(failure_model="drift", injection_interval=5200),
        )  # must not raise - two distinct fault types on the same schedule is the whole point

    def test_rejects_secondary_fault_with_same_failure_model_as_primary(self):
        with pytest.raises(ValueError, match="secondary_fault.failure_model must differ"):
            ExperimentConfig(
                name="x",
                fault=FaultConfig(failure_model="density"),
                secondary_fault=FaultConfig(failure_model="density"),
            )

    def test_rejects_secondary_fault_with_mismatched_injection_interval(self):
        with pytest.raises(ValueError, match="secondary_fault.injection_interval"):
            ExperimentConfig(
                name="x",
                fault=FaultConfig(failure_model="density", injection_interval=5200),
                secondary_fault=FaultConfig(failure_model="drift", injection_interval=100),
            )


class TestWaveDensitiesValidation:
    def test_rejects_empty_list(self):
        with pytest.raises(ValueError, match="wave_densities must be non-empty"):
            FaultConfig(wave_densities=[])

    @pytest.mark.parametrize("bad", [0.0, -0.1, 1.1])
    def test_rejects_out_of_range_entry(self, bad):
        with pytest.raises(ValueError, match=r"wave_densities entries must each be in \(0, 1\]"):
            FaultConfig(wave_densities=[0.1, bad])

    def test_accepts_valid_schedule(self):
        FaultConfig(wave_densities=[0.05, 0.15, 1.0])


class TestRetrainingConfigValidation:
    @pytest.mark.parametrize(
        "field,value,match",
        [
            ("policy", "sometimes", "Unknown RetrainingConfig.policy"),
            ("reference_mode", "stale", "Unknown RetrainingConfig.reference_mode"),
            ("remap_order", "alphabetical", "Unknown RetrainingConfig.remap_order"),
            ("remap_pacing", "lazy", "Unknown RetrainingConfig.remap_pacing"),
            ("burst_lr", 0.0, "burst_lr must be positive"),
            ("burst_lr", -1e-3, "burst_lr must be positive"),
            ("burst_batches", -1, "burst_batches must be >= 0"),
            ("burst_t_max", 0, "burst_t_max must be >= 1"),
            ("patience_window", 0, "patience_window must be >= 1"),
            ("patience_eval_every", 0, "patience_eval_every must be >= 1"),
            ("patience_min_gain", -0.1, "patience_min_gain must be >= 0"),
            ("canary_gate_delta", -0.1, "canary_gate_delta must be >= 0"),
            ("canary_size", 0, "canary_size must be >= 1"),
            ("bn_recal_batches", -1, "bn_recal_batches must be >= 0"),
            ("expected_waves", 0, "expected_waves must be >= 1"),
            ("fisher_ema_decay", 1.0, "fisher_ema_decay must be in"),
            ("fisher_ema_decay", -0.1, "fisher_ema_decay must be in"),
        ],
    )
    def test_rejects_invalid_field(self, field, value, match):
        with pytest.raises(ValueError, match=match):
            RetrainingConfig(**{field: value})

    @pytest.mark.parametrize("policy", ["static", "periodic", "capacity"])
    def test_fixed_length_burst_policies_need_positive_burst_batches(self, policy):
        with pytest.raises(ValueError, match="burst_batches must be >= 1"):
            RetrainingConfig(policy=policy, burst_batches=0)

    @pytest.mark.parametrize("policy", ["none", "static", "periodic", "capacity", "co_mitigation"])
    def test_accepts_every_policy(self, policy):
        RetrainingConfig(
            policy=policy, reference_mode="living" if policy == "co_mitigation" else "frozen"
        )

    def test_co_mitigation_requires_living_reference(self):
        with pytest.raises(ValueError, match="requires reference_mode='living'"):
            RetrainingConfig(policy="co_mitigation")
        RetrainingConfig(policy="co_mitigation", reference_mode="living", burst_mode="fixed")

    def test_burst_mode_vocabulary_and_fixed_needs_batches(self):
        with pytest.raises(ValueError, match="Unknown RetrainingConfig.burst_mode"):
            RetrainingConfig(burst_mode="sometimes")
        with pytest.raises(ValueError, match="burst_batches must be >= 1"):
            RetrainingConfig(
                policy="co_mitigation", reference_mode="living", burst_mode="fixed", burst_batches=0
            )

    def test_priors_path_only_for_co_mitigation(self):
        with pytest.raises(ValueError, match="priors_path is only read by"):
            RetrainingConfig(policy="capacity", priors_path="p.json")
        RetrainingConfig(policy="co_mitigation", reference_mode="living", priors_path="p.json")

    def test_legacy_enable_fault_aware_with_default_policy_is_valid(self):
        ExperimentConfig(name="x", retraining=RetrainingConfig(enable_fault_aware=True))


class TestRegimeValidation:
    def _deploy(self, **kw):
        base = {
            "name": "x",
            "regime": "deployment",
            "pre_deploy_checkpoint": "ckpt.pt",
            "fault": FaultConfig(wave_densities=[0.1, 0.1]),
        }
        base.update(kw)
        return ExperimentConfig(**base)

    def test_rejects_unknown_regime(self):
        with pytest.raises(ValueError, match="Unknown ExperimentConfig.regime"):
            ExperimentConfig(name="x", regime="sometimes")

    def test_accepts_valid_deployment(self):
        self._deploy()

    def test_deployment_requires_checkpoint(self):
        with pytest.raises(ValueError, match="requires pre_deploy_checkpoint"):
            self._deploy(pre_deploy_checkpoint=None)

    def test_deployment_requires_wave_densities(self):
        with pytest.raises(ValueError, match="requires fault.wave_densities"):
            self._deploy(fault=FaultConfig())

    def test_deployment_requires_density_failure_model(self):
        with pytest.raises(ValueError, match="failure_model='density'"):
            self._deploy(
                fault=FaultConfig(
                    failure_model="exponential", exponential_rate=1e-6, wave_densities=[0.1]
                )
            )

    def test_predeploy_needs_neither_checkpoint_nor_waves(self):
        ExperimentConfig(name="x", regime="deployment", save_pre_deploy_checkpoint=True)

    def test_predeploy_and_checkpoint_are_mutually_exclusive(self):
        with pytest.raises(ValueError, match="mutually exclusive"):
            self._deploy(save_pre_deploy_checkpoint=True)

    def test_save_pre_deploy_requires_deployment_regime(self):
        with pytest.raises(ValueError, match="requires regime='deployment'"):
            ExperimentConfig(name="x", save_pre_deploy_checkpoint=True)

    def test_continuous_forbids_wave_densities(self):
        with pytest.raises(ValueError, match="only used by regime='deployment'"):
            ExperimentConfig(name="x", fault=FaultConfig(wave_densities=[0.1]))

    def test_continuous_forbids_pre_deploy_checkpoint(self):
        with pytest.raises(ValueError, match="pre_deploy_checkpoint is only used"):
            ExperimentConfig(name="x", pre_deploy_checkpoint="ckpt.pt")

    def test_deployment_forbids_legacy_fault_aware(self):
        with pytest.raises(ValueError, match="legacy continuous-regime"):
            self._deploy(retraining=RetrainingConfig(enable_fault_aware=True))

    def test_legacy_flag_and_policy_are_mutually_exclusive(self):
        # policy="capacity" is wired into the continuous regime, so validation gets as far as
        # the legacy-flag/policy exclusivity check (policy="static" is rejected earlier). The
        # continuous-regime policy guard needs predictive dispatch off.
        with pytest.raises(ValueError, match="mutually exclusive"):
            ExperimentConfig(
                name="x",
                mitigation=MitigationConfig(enable_predictive=False),
                retraining=RetrainingConfig(enable_fault_aware=True, policy="capacity"),
            )

    def test_living_reference_requires_idealized_per_cell(self):
        with pytest.raises(ValueError, match="reference_mode='living' requires"):
            ExperimentConfig(
                name="x",
                crossbar=CrossbarConfig(redundancy_scheme="row_col_granularity"),
                retraining=RetrainingConfig(reference_mode="living"),
            )

    def test_living_reference_with_idealized_per_cell_is_valid(self):
        ExperimentConfig(name="x", retraining=RetrainingConfig(reference_mode="living"))

    def test_paced_continuous_needs_expected_waves(self):
        with pytest.raises(ValueError, match="needs expected_waves"):
            ExperimentConfig(name="x", retraining=RetrainingConfig(remap_pacing="paced"))

    def test_paced_continuous_with_expected_waves_is_valid(self):
        ExperimentConfig(
            name="x", retraining=RetrainingConfig(remap_pacing="paced", expected_waves=3)
        )

    def test_co_mitigation_allowed_in_both_regimes(self):
        # continuous-regime policies replace health-gated dispatch: predictive must be off
        mit = MitigationConfig(enable_remapping=True, enable_predictive=False)
        retr = RetrainingConfig(policy="co_mitigation", reference_mode="living")
        ExperimentConfig(name="x", mitigation=mit, retraining=retr)
        self._deploy(mitigation=mit, retraining=retr)

    def test_co_mitigation_needs_remapping_and_per_cell_spares(self):
        retr = RetrainingConfig(policy="co_mitigation", reference_mode="living")
        with pytest.raises(ValueError, match="needs mitigation.enable_remapping"):
            ExperimentConfig(
                name="x",
                mitigation=MitigationConfig(enable_remapping=False, enable_predictive=False),
                retraining=retr,
            )
        with pytest.raises(ValueError, match="reference_mode='living' requires"):
            ExperimentConfig(
                name="x",
                mitigation=MitigationConfig(enable_remapping=True, enable_predictive=False),
                crossbar=CrossbarConfig(redundancy_scheme="row_col_granularity"),
                retraining=retr,
            )
