from neurofault.config import load_config


def test_load_config_parses_nested_dataclasses(tmp_path):
    yaml_path = tmp_path / "test.yaml"
    yaml_path.write_text(
        """
name: test_arm
epochs: 3
mitigation:
  enable_soft_mitigation: true
  enable_layer_reset: false
  health_threshold: 60.0
fault:
  density: 0.2
"""
    )

    config = load_config(yaml_path)

    assert config.name == "test_arm"
    assert config.epochs == 3
    assert config.mitigation.enable_soft_mitigation is True
    assert config.mitigation.enable_layer_reset is False
    assert config.mitigation.health_threshold == 60.0
    assert config.fault.density == 0.2
    # Fields not present in the YAML must keep their dataclass defaults.
    assert config.mitigation.enable_remapping is False
    assert config.crossbar.patch_layer_types == ["Linear", "Conv2d"]
    # Not present in the YAML at all - must keep its dataclass default too.
    assert config.retraining.enable_fault_aware is False
    assert config.model_arch == "resnet9"  # default: every existing config is the real network


def test_load_config_parses_model_arch(tmp_path):
    yaml_path = tmp_path / "test.yaml"
    yaml_path.write_text("name: t\nmodel_arch: tiny_cnn\n")
    assert load_config(yaml_path).model_arch == "tiny_cnn"


def test_load_config_parses_retraining_section(tmp_path):
    yaml_path = tmp_path / "test.yaml"
    yaml_path.write_text(
        """
name: test_arm
retraining:
  enable_fault_aware: true
"""
    )

    config = load_config(yaml_path)

    assert config.retraining.enable_fault_aware is True
