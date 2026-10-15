"""No coverage of devices/registry.py existed before this - a real,
pre-existing gap closed alongside adding the fefet/sttmram device names.
"""

from __future__ import annotations

import pytest

from neurofault.config import DeviceConfig
from neurofault.devices.registry import build_device


def test_crosssim_device_ignores_name_uses_raw_r_on_r_off():
    device_cls, params = build_device(
        DeviceConfig(name="anything", r_on=50.0, r_off=5000.0), simulator="crosssim"
    )
    assert device_cls is None
    assert params["r_on"] == 50.0
    assert params["r_off"] == 5000.0


def test_crosssim_device_folds_in_extra_params():
    _device_cls, params = build_device(
        DeviceConfig(name="anything", extra_params={"read_noise": 0.02}), simulator="crosssim"
    )
    assert params["read_noise"] == 0.02


@pytest.mark.parametrize(
    "name,expected_cls_name",
    [
        ("pcm", "PCMPresetDevice"),
        ("reram_es", "ReRamESPresetDevice"),
        ("reram_sb", "ReRamSBPresetDevice"),
        ("ecram", "EcRamPresetDevice"),
        ("fefet", "FeFETPresetDevice"),
        ("sttmram", "STTMRAMPresetDevice"),
    ],
)
def test_aihwkit_device_resolves_every_registered_name(name, expected_cls_name):
    pytest.importorskip("aihwkit", reason="AIHWKit not installed (uv sync --extra aihwkit)")

    device_cls, params = build_device(DeviceConfig(name=name), simulator="aihwkit")

    assert device_cls.__name__ == expected_cls_name
    assert isinstance(params, dict)


def test_aihwkit_device_raises_on_unknown_name():
    pytest.importorskip("aihwkit", reason="AIHWKit not installed (uv sync --extra aihwkit)")

    with pytest.raises(ValueError, match="Unknown aihwkit device name"):
        build_device(DeviceConfig(name="not_a_real_device"), simulator="aihwkit")


def test_build_device_raises_on_unknown_simulator():
    with pytest.raises(ValueError, match="Unknown simulator backend"):
        build_device(DeviceConfig(), simulator="not_a_real_backend")
