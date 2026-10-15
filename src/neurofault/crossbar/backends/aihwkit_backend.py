"""AIHWKit backend: replaces each patched Linear/Conv2d layer with a real
AIHWKit analog equivalent whose forward pass computes through the same
per-tile matrix this module's ConductanceAccessor reads and writes.

Root cause found and worked around during development: AIHWKit's default tile
(`SingleRPUConfig`, backed by its compiled RPU C++ extension) is broken -
`AnalogLinear`/`AnalogConv2d` construction fails with `RuntimeError: Invalid
weights dimensions` on every platform tested (macOS arm64, Linux x86_64),
every device preset, with or without bias. Confirmed the bug lives in the
compiled binding itself, not the nn.Module wrapper: calling the lowest-level
tile's set_weights() directly (bypassing from_digital()/reset_parameters()
entirely) reproduces the identical failure - so there is no lower-level way
to route around it. AIHWKit ships a second, pure-PyTorch tile implementation
- `TorchInferenceRPUConfig` (tile_class=TorchInferenceTile,
simulator_tile_class=TorchSimulatorTile) - that never touches the broken
binding. Verified hands-on: from_digital() works for both layer types, a
real forward pass computes through the tile (a written fault measurably
changes output), and the existing device-preset classes already used by
devices/registry.py (PCMPresetDevice, ReRamESPresetDevice, etc.) work
unchanged when assigned as `rpu_config.device = preset_cls()` *after*
construction (passing `device=` as a TorchInferenceRPUConfig() constructor
kwarg raises TypeError - this config class takes the device post-init).

Unit-space caveat, same as crosssim_backend.py: AIHWKit's tile.get_weights()/
set_weights() operate in the layer's own weight-space (not literal Ohms/
Siemens), and that range is signed - so build_handles() below derives this
handle's r_on/r_off from the tile's weight *magnitude* (symmetric
[-magnitude, +magnitude] as [g_min, g_max]) rather than its raw signed
bounds, for the exact same reason crosssim_backend.py does: deriving r_off
from a negative raw bound would flip HealthMonitor's g_min/g_max ordering
for an all-negative-weight layer, since 1/x is order-reversing for negative
x. See crosssim_backend.py's module docstring for the fuller reasoning.

Bias handling differs from CrossSim, though: AIHWKit's tile.set_weights(weight,
bias)/get_weights() always take/return bias together - there is no
"digital bias" toggle at the tile level the way CrossSim's bias_rows=0
keeps bias out of the accessor entirely. So AIHWKitAccessor caches the
layer's bias once at construction and passes it through unchanged on every
write() - mitigation code only ever supplies a weight-shaped matrix, same as
every other backend's ConductanceAccessor.write(matrix) signature.

crossbar_config.tile_shape IS wired, via rpu_config.mapping.max_output_size/
max_input_size - intuitively named (verified: max_output_size=4 on a
Linear(20,16) layer produced 4 tiles of shape (4,20), i.e. constrains
out_features as the name suggests - unlike CrossSim's inverted rows_max/
cols_max, see crosssim_backend.py). These already default to 512/512 (not
unlimited) - this backend has been silently tiling all along whenever a
layer exceeds 512 in either dimension; _build_rpu_config() below just makes
that an explicit, configured value instead of AIHWKit's hardcoded default.
build_handles() already iterates analog_tiles() per-tile, so multi-tile
layers were already handled correctly before this - no further changes
needed there. mapping_routine_for() below still only covers the common
single-tile-per-layer case, same conservative stance as the other backends -
a multi-tile layer leaves layer_reset unavailable (system.py already
degrades gracefully) rather than risk a wrong reset.

ADC/DAC resolution (crossbar_config.adc_resolution/dac_resolution) IS wired:
TorchInferenceRPUConfig().forward (IOParameters) already defaults to
non-ideal quantization out of the box - inp_res/out_res are resolution step
sizes, not bit counts, following the formula res = 1/(2**bits - 2) (verified
by matching AIHWKit's own defaults: inp_res=0.00794=1/(2**7-2), i.e. its
un-configured default is already a 7-bit DAC; out_res=0.00196=1/(2**9-2), a
9-bit ADC). _build_rpu_config() below applies that same formula from our own
config values instead of leaving AIHWKit's hardcoded defaults in place.

crossbar_config.circuit_topology (1T1R/0T1R/1S1R) is NOT supported on
this backend - real, verified constraint, not a wiring gap. AIHWKit's
IOParameters does expose real fields for exactly this (forward.r_series,
forward.ir_drop/ir_drop_g_ratio), and crosssim_backend.py wires the CrossSim
equivalent successfully - but TorchInferenceTile (this backend's only
working tile, see above: the default SingleRPUConfig tile is broken)
explicitly refuses both: AIHWKit's own analog_mvm.py::check_support() raises
`ConfigError("IR drop not supported in torch tile")` and
`ConfigError("Voltage offset or R-series not supported in torch tile")`,
confirmed by direct construction attempts. _build_rpu_config() below raises a
clear error if circuit_topology is set for this backend rather than silently
ignoring it - use simulator="crosssim" for this axis.

crossbar_config.devices_per_synapse (multi-device/bit-slicing) IS
wired: when >1, the resolved device class is wrapped as
`VectorUnitCell(unit_cell_devices=[device_cls() for _ in range(n)])` instead
of assigned directly as `device_cls()`. Verified directly: VectorUnitCell
constructs fine under TorchInferenceRPUConfig, a fault write still changes
the forward pass, and its default update_policy=ALL already lets a normal
optimizer.step() reach every constituent device with no extra
synchronize() call - confirmed with a real multi-step Adam probe (a first,
looser torch.allclose check gave a false negative; re-verified with an
explicit weight-diff before trusting it as real, not a repeat of this
project's CrossSim training-sync bug).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import torch

from neurofault.config import CrossbarConfig
from neurofault.crossbar.array import CrossbarHandle

logger = logging.getLogger(__name__)

_LAYER_TYPES = {
    "Linear": torch.nn.Linear,
    "Conv2d": torch.nn.Conv2d,
}


@dataclass
class AIHWKitAccessor:
    tile: object  # a per-tile AIHWKit analog tile, from layer.analog_tiles()
    bias: torch.Tensor | None  # cached once; passed through unchanged on every write()

    @property
    def shape(self) -> tuple[int, int]:
        weight, _ = self.tile.get_weights()
        return tuple(weight.shape)

    def read(self) -> torch.Tensor:
        weight, _ = self.tile.get_weights()
        return weight

    def write(self, matrix: torch.Tensor) -> None:
        self.tile.set_weights(matrix, self.bias)


def _build_rpu_config(
    device_cls: type | None, device_params: dict, crossbar_config: CrossbarConfig
):
    from aihwkit.simulator.configs import TorchInferenceRPUConfig

    rpu_config = TorchInferenceRPUConfig()
    if device_cls is not None:
        if crossbar_config.devices_per_synapse > 1:
            from aihwkit.simulator.configs.compounds import VectorUnitCell

            rpu_config.device = VectorUnitCell(
                unit_cell_devices=[device_cls() for _ in range(crossbar_config.devices_per_synapse)]
            )
        else:
            rpu_config.device = device_cls()

    # ADC/DAC resolution: see module docstring for the res = 1/(2**bits - 2)
    # formula, verified against AIHWKit's own IOParameters defaults. Clamp to
    # a minimum of 2 bits - below that the formula divides by zero or goes
    # negative, and no real ADC/DAC is 0/1-bit anyway.
    dac_bits = max(crossbar_config.dac_resolution, 2)
    adc_bits = max(crossbar_config.adc_resolution, 2)
    rpu_config.forward.inp_res = 1.0 / (2**dac_bits - 2)
    rpu_config.forward.out_res = 1.0 / (2**adc_bits - 2)

    if crossbar_config.tile_shape is not None:
        tile_rows, tile_cols = crossbar_config.tile_shape
        rpu_config.mapping.max_output_size = tile_rows
        rpu_config.mapping.max_input_size = tile_cols

    if crossbar_config.circuit_topology is not None:
        # See module docstring: TorchInferenceTile (the only working tile on
        # this backend - the default SingleRPUConfig tile is broken, see
        # above) explicitly refuses both of the knobs that would model this
        # (AIHWKit's own check_support() raises ConfigError for r_series and
        # for ir_drop, verified directly) - not a wiring gap, a real,
        # verified constraint of the tile this project depends on. Fail
        # loud rather than silently ignore the config value.
        raise ValueError(
            "circuit_topology is not supported on the aihwkit backend: "
            "TorchInferenceTile (this backend's only working tile - see "
            "module docstring) explicitly rejects both r_series and ir_drop "
            "(AIHWKit's own ConfigError). Use simulator='crosssim' for "
            "circuit_topology, or leave it unset (None) here."
        )

    return rpu_config


def patch_layers(
    model: torch.nn.Module,
    crossbar_config: CrossbarConfig,
    device_cls: type | None,
    device_params: dict,
    calibrated_ranges: dict[str, tuple[float, float]] | None = None,
) -> torch.nn.Module:
    """Replaces only the configured layer types (crossbar_config.patch_layer_types)
    with AIHWKit analog equivalents via from_digital() - an explicit per-type
    filter, same pattern as crosssim_backend.py's patch_layers(), rather than
    converting every layer type AIHWKit happens to support.

    calibrated_ranges: accepted only for interface symmetry with
    crosssim_backend.patch_layers() (system.py calls both backends' patch_layers
    identically) - CrossSim-only (its ADCRangeLimits.CALIBRATED fix); AIHWKit has no equivalent ADC-range concept to calibrate, so this is
    always None for this backend (system.py never sets config.crossbar.
    calibrate_adc's calibration path for a non-CrossSim backend).
    """
    from aihwkit.nn import AnalogConv2d, AnalogLinear

    conversion_map = {
        torch.nn.Linear: AnalogLinear,
        torch.nn.Conv2d: AnalogConv2d,
    }
    patch_targets = tuple(_LAYER_TYPES[t] for t in crossbar_config.patch_layer_types)
    rpu_config = _build_rpu_config(device_cls, device_params, crossbar_config)

    def _convert_children(module: torch.nn.Module) -> None:
        for name, child in list(module.named_children()):
            if isinstance(child, patch_targets) and type(child) in conversion_map:
                analog_cls = conversion_map[type(child)]
                setattr(module, name, analog_cls.from_digital(child, rpu_config))
            else:
                _convert_children(child)

    _convert_children(model)
    return model


def build_handles(
    patched_model: torch.nn.Module, device_params: dict, crossbar_config: CrossbarConfig
) -> dict[str, CrossbarHandle]:
    from aihwkit.nn import AnalogConv2d, AnalogLinear

    analog_types = (AnalogLinear, AnalogConv2d)
    handles: dict[str, CrossbarHandle] = {}

    for name, module in patched_model.named_modules():
        if not isinstance(module, analog_types):
            continue

        for i, tile in enumerate(module.analog_tiles()):
            weight, bias = tile.get_weights()

            # See module docstring: derive this handle's r_on/r_off from the
            # tile's actual weight-space magnitude, not device_params'
            # literal Ohms, so HealthMonitor/faults/injection.py normalize
            # against the range this accessor's read()/write() actually
            # operate in.
            magnitude = max(float(weight.min().abs()), float(weight.max().abs()), 1e-12)
            handle_params = dict(device_params)
            handle_params["r_on"] = 1.0 / magnitude
            handle_params["r_off"] = -1.0 / magnitude

            handle_name = f"{name}_crossbar_{i}"
            handles[handle_name] = CrossbarHandle(
                name=handle_name,
                accessor=AIHWKitAccessor(tile=tile, bias=bias),
                layer=module,
                device_params=handle_params,
            )

    logger.info("Built %d AIHWKit-backed crossbar handle(s)", len(handles))
    return handles


def mapping_routine_for(scheme: str):
    """Returns the weight->conductance mapping `layer_reset` needs. AIHWKit's
    tile.set_weights() takes real weight-space values directly - no separate
    weight->conductance derivation needed (mirrors crosssim_backend.py's own
    mapping_routine_for). `scheme` is accepted only so every backend's
    mapping_routine_for() shares one call signature; AIHWKit has no
    DoubleColumn-style split to honor.
    """

    def _map(weight: torch.Tensor, r_on: float, r_off: float, scheme: str) -> torch.Tensor:
        rows = weight.shape[0]
        return weight.reshape(rows, -1)

    return _map


def synchronize(model: torch.nn.Module) -> None:
    """No-op: verified during development that AIHWKit's analog tiles already
    reflect optimizer.step() updates without any extra call (unlike
    CrossSim, see crosssim_backend.py's synchronize() and module docstring
    for that backend's training gap) - AIHWKit is purpose-built for
    hardware-aware training, so this isn't needed here. Exists so system.py
    can call backend.synchronize(model) unconditionally regardless of which
    backend is active."""
