"""Maps a device-technology name to its simulator-specific device class and
default parameters, dispatched per simulation backend (crosssim | aihwkit).
Replaces the VTEAM-only block that was duplicated verbatim in two places in
the original codebase - device AND backend choice are now config values, not
hardcoded if-statements.

Each backend library is imported lazily inside its own build_*_device()
function, so this module stays importable without any of them installed.
"""

from __future__ import annotations

from neurofault.config import DeviceConfig


def build_device(config: DeviceConfig, simulator: str = "crosssim") -> tuple[type, dict]:
    """Returns (device class, params dict) for the given device config and backend."""
    if simulator == "crosssim":
        return _build_crosssim_device(config)
    if simulator == "aihwkit":
        return _build_aihwkit_device(config)
    raise ValueError(f"Unknown simulator backend: {simulator!r}")


def _build_crosssim_device(config: DeviceConfig) -> tuple[type, dict]:
    """CrossSim has no distinct swappable "device class" the way MemTorch/
    AIHWKit do - it's one generic RRAM-like DeviceParameters model configured
    by fields (Rmin/Rmax and friends), not named presets. crosssim_backend.py
    reads r_on/r_off straight out of the returned params dict to build its
    CrossSimParameters; there is no device class to return.

    Real, structural limitation, not papered over: this means CrossSim cannot
    differentiate technology-specific switching dynamics at all (e.g. FeFET's
    near-linear symmetric updates vs. PCM's drift) - only the conductance
    range (r_on/r_off) differs by technology on this backend. See
    experiments/configs/device_fefet_crosssim.yaml/device_sttmram_crosssim.yaml
    for real, paper-derived r_on/r_off values used to at least exercise these
    technologies' dynamic range on this backend. The richer, per-technology
    device presets (below) exist only for the aihwkit backend, which has an
    actual swappable device model."""
    params = {"r_on": config.r_on, "r_off": config.r_off, **config.extra_params}
    return None, params


def _build_aihwkit_device(config: DeviceConfig) -> tuple[type, dict]:
    """AIHWKit presets (aihwkit.simulator.presets): calibrated against real
    IBM hardware measurements (1M-device PCM array; ReRAM/HfOx arrays) -
    the richest real-silicon-grounded device statistics of the two backends,
    but with no persistent fault model of its own (noise/drift only) - fault
    injection for this backend is entirely neurofault's own custom path.
    These preset classes are assigned onto crossbar/backends/aihwkit_backend.py's
    TorchInferenceRPUConfig post-construction (`rpu_config.device = preset_cls()`),
    not passed to a constructor kwarg - see that module's docstring for why
    (AIHWKit's default SingleRPUConfig tile is broken upstream; TorchInferenceRPUConfig
    is the pure-PyTorch tile that actually works, verified during development)."""
    from aihwkit.simulator.presets import devices as aihwkit_devices

    from neurofault.devices import presets as neurofault_devices

    name_map = {
        "pcm": (aihwkit_devices, "PCMPresetDevice"),
        "reram_es": (aihwkit_devices, "ReRamESPresetDevice"),
        "reram_sb": (aihwkit_devices, "ReRamSBPresetDevice"),
        "ecram": (aihwkit_devices, "EcRamPresetDevice"),
        # Ferroelectric/Magnetic - no AIHWKit preset exists for either
        # (confirmed via dir(aihwkit.simulator.presets.devices)), so these are
        # neurofault's own, in devices/presets.py.
        "fefet": (neurofault_devices, "FeFETPresetDevice"),
        "sttmram": (neurofault_devices, "STTMRAMPresetDevice"),
    }
    if config.name not in name_map:
        raise ValueError(
            f"Unknown aihwkit device name: {config.name!r} (expected one of {list(name_map)})"
        )

    module, cls_name = name_map[config.name]
    preset_cls = getattr(module, cls_name)
    return preset_cls, dict(config.extra_params)
