import pytest
import torch

from neurofault.config import CrossbarConfig
from neurofault.crossbar.backends.aihwkit_backend import mapping_routine_for
from neurofault.mitigation.reset import apply_layer_reset
from tests.unit.fakes import FakeCrossbar

pytest.importorskip("aihwkit", reason="AIHWKit not installed (uv sync --extra aihwkit)")

from neurofault.crossbar.backends.aihwkit_backend import build_handles, patch_layers, synchronize


def test_patch_layers_respects_patch_layer_types():
    model = torch.nn.Sequential(torch.nn.Linear(6, 4), torch.nn.Conv2d(3, 2, kernel_size=3))
    crossbar_config = CrossbarConfig(patch_layer_types=["Linear"])

    patched = patch_layers(model, crossbar_config, None, {"r_on": 100.0, "r_off": 10000.0})
    handles = build_handles(patched, {"r_on": 100.0, "r_off": 10000.0}, crossbar_config)

    assert len(handles) == 1  # only the Linear layer got converted


def test_build_handles_read_write_round_trip():
    model = torch.nn.Sequential(torch.nn.Linear(6, 4))
    crossbar_config = CrossbarConfig(patch_layer_types=["Linear"])
    device_params = {"r_on": 100.0, "r_off": 10000.0}

    patched = patch_layers(model, crossbar_config, None, device_params)
    handles = build_handles(patched, device_params, crossbar_config)
    handle = next(iter(handles.values()))

    original = handle.accessor.read()
    assert original.shape == (4, 6)

    mutated = original.clone()
    mutated[0, 0] = mutated.max().item() * 100
    handle.accessor.write(mutated)

    assert torch.allclose(handle.accessor.read(), mutated)


def test_write_preserves_cached_bias():
    """Decision 5: AIHWKit's tile.set_weights(weight, bias) always takes both
    together, unlike CrossSim - the accessor must cache bias once and reuse
    it on every write() so mitigation code never has to supply it."""
    model = torch.nn.Sequential(torch.nn.Linear(6, 4, bias=True))
    crossbar_config = CrossbarConfig(patch_layer_types=["Linear"])
    device_params = {"r_on": 100.0, "r_off": 10000.0}

    patched = patch_layers(model, crossbar_config, None, device_params)
    handles = build_handles(patched, device_params, crossbar_config)
    handle = next(iter(handles.values()))

    original_bias = handle.accessor.bias.clone()

    mutated = handle.accessor.read().clone()
    mutated[0, 0] = mutated.max().item() * 100
    handle.accessor.write(mutated)

    _, bias_after = handle.accessor.tile.get_weights()
    assert torch.allclose(bias_after, original_bias)


def test_fault_write_changes_forward_pass_output():
    """Writing a fault into the accessor must change the model's real
    forward output - the whole point of this backend."""
    model = torch.nn.Sequential(torch.nn.Linear(6, 4))
    crossbar_config = CrossbarConfig(patch_layer_types=["Linear"])
    device_params = {"r_on": 100.0, "r_off": 10000.0}

    patched = patch_layers(model, crossbar_config, None, device_params)
    handles = build_handles(patched, device_params, crossbar_config)
    handle = next(iter(handles.values()))

    x = torch.randn(2, 6)
    output_before = patched(x).detach().clone()

    matrix = handle.accessor.read().clone()
    matrix[0, 0] = matrix.max().item() * 100
    handle.accessor.write(matrix)

    output_after = patched(x).detach().clone()
    assert not torch.allclose(output_before, output_after)


def test_reduced_adc_dac_resolution_still_produces_working_fault_to_accuracy_forward():
    """Non-default ADC/DAC quantization must not silently break the
    forward pass or the fault-to-accuracy property this backend exists for."""
    model = torch.nn.Sequential(torch.nn.Linear(6, 4))
    crossbar_config = CrossbarConfig(
        patch_layer_types=["Linear"], adc_resolution=4, dac_resolution=4
    )
    device_params = {"r_on": 100.0, "r_off": 10000.0}

    patched = patch_layers(model, crossbar_config, None, device_params)
    handles = build_handles(patched, device_params, crossbar_config)
    handle = next(iter(handles.values()))

    x = torch.randn(2, 6)
    output_before = patched(x).detach().clone()

    matrix = handle.accessor.read().clone()
    matrix[0, 0] = matrix.max().item() * 100
    handle.accessor.write(matrix)

    output_after = patched(x).detach().clone()
    assert not torch.allclose(output_before, output_after)


def test_tile_shape_produces_expected_multi_tile_partitioning():
    """AIHWKit's max_output_size/max_input_size are intuitively named
    (unlike CrossSim's inverted rows_max/cols_max) - verified during development.
    Asserts the actual per-tile shapes, not just tile count."""
    model = torch.nn.Sequential(torch.nn.Linear(20, 16))  # weight shape (16, 20)
    crossbar_config = CrossbarConfig(patch_layer_types=["Linear"], tile_shape=(4, 20))
    device_params = {"r_on": 100.0, "r_off": 10000.0}

    patched = patch_layers(model, crossbar_config, None, device_params)
    handles = build_handles(patched, device_params, crossbar_config)

    # tile_shape=(4, 20): max_output_size=4 -> ceil(16/4)=4 tiles, max_input_size=20
    # leaves the 20 input columns in one chunk per tile.
    assert len(handles) == 4
    for handle in handles.values():
        assert handle.accessor.shape == (4, 20)


def test_circuit_topology_is_not_supported_on_this_backend():
    """Real, verified constraint (see module docstring), not a wiring gap:
    TorchInferenceTile - this backend's only working tile - explicitly
    refuses both r_series and ir_drop via AIHWKit's own check_support()."""
    model = torch.nn.Sequential(torch.nn.Linear(6, 4))
    crossbar_config = CrossbarConfig(patch_layer_types=["Linear"], circuit_topology="1T1R")
    device_params = {"r_on": 100.0, "r_off": 10000.0}

    with pytest.raises(ValueError, match="circuit_topology is not supported"):
        patch_layers(model, crossbar_config, None, device_params)


def test_devices_per_synapse_wraps_in_vector_unit_cell_and_still_trains():
    from neurofault.devices.presets import FeFETPresetDevice

    model = torch.nn.Sequential(torch.nn.Linear(6, 4))
    crossbar_config = CrossbarConfig(patch_layer_types=["Linear"], devices_per_synapse=3)
    device_params = {"r_on": 100.0, "r_off": 10000.0}

    patched = patch_layers(model, crossbar_config, FeFETPresetDevice, device_params)
    handles = build_handles(patched, device_params, crossbar_config)
    handle = next(iter(handles.values()))

    assert handle.accessor.shape == (4, 6)  # logical shape unchanged by combining devices

    x = torch.randn(2, 6)
    output_before = patched(x).detach().clone()
    matrix = handle.accessor.read().clone()
    matrix[0, 0] = matrix.max().item() * 100
    handle.accessor.write(matrix)
    assert not torch.allclose(output_before, patched(x).detach().clone())

    matrix_before_training = handle.accessor.read().clone()
    optimizer = torch.optim.Adam(patched.parameters(), lr=0.5)
    for _ in range(5):
        optimizer.zero_grad()
        patched(x).sum().backward()
        optimizer.step()

    assert not torch.allclose(matrix_before_training, handle.accessor.read())


def test_optimizer_step_alone_already_reaches_the_analog_tile():
    """Verified during development: unlike CrossSim, AIHWKit's tiles already
    reflect optimizer.step() updates without any extra call - synchronize()
    is a documented no-op here, not a missing feature."""
    model = torch.nn.Sequential(torch.nn.Linear(6, 4))
    crossbar_config = CrossbarConfig(patch_layer_types=["Linear"])
    device_params = {"r_on": 100.0, "r_off": 10000.0}
    patched = patch_layers(model, crossbar_config, None, device_params)
    lin = patched[0]
    tile = next(lin.analog_tiles())

    weight_before, _ = tile.get_weights()
    weight_before = weight_before.clone()
    optimizer = torch.optim.Adam(patched.parameters(), lr=0.1)
    x = torch.randn(4, 6)
    for _ in range(5):
        optimizer.zero_grad()
        patched(x).sum().backward()
        optimizer.step()

    weight_after, _ = tile.get_weights()
    assert not torch.allclose(weight_before, weight_after)


def test_synchronize_is_a_safe_noop():
    model = torch.nn.Sequential(torch.nn.Linear(6, 4))
    crossbar_config = CrossbarConfig(patch_layer_types=["Linear"])
    device_params = {"r_on": 100.0, "r_off": 10000.0}
    patched = patch_layers(model, crossbar_config, None, device_params)

    synchronize(patched)  # must not raise


def test_build_handles_derives_symmetric_r_on_r_off_from_weight_magnitude():
    model = torch.nn.Sequential(torch.nn.Linear(6, 4))
    crossbar_config = CrossbarConfig(patch_layer_types=["Linear"])
    device_params = {"r_on": 100.0, "r_off": 10000.0}

    patched = patch_layers(model, crossbar_config, None, device_params)
    handles = build_handles(patched, device_params, crossbar_config)
    handle = next(iter(handles.values()))

    g_min = 1.0 / handle.device_params["r_off"]
    g_max = 1.0 / handle.device_params["r_on"]
    assert g_min < g_max  # ordering must survive the signed weight-space range
    assert g_min == -g_max  # symmetric around 0, per module docstring


def test_mapping_routine_is_identity_reshape():
    mapping_routine = mapping_routine_for("DoubleColumn")
    weight = torch.tensor([[-2.0, 0.0], [1.0, 3.0]])

    matrix = mapping_routine(weight, r_on=100.0, r_off=10000.0, scheme="DoubleColumn")

    assert torch.equal(matrix, weight)


def test_mapping_routine_flattens_conv2d_shaped_weights():
    mapping_routine = mapping_routine_for("DoubleColumn")
    weight = torch.randn(4, 3, 3, 3)  # Conv2d weight: [out_channels, in_channels, kh, kw]

    matrix = mapping_routine(weight, r_on=100.0, r_off=10000.0, scheme="DoubleColumn")

    assert matrix.shape == (4, 27)


def test_layer_reset_works_end_to_end_with_aihwkit_mapping_routine():
    layer = torch.nn.Linear(4, 4)
    original_weight = layer.weight.data.clone()
    layer.weight.data.fill_(999.0)  # simulate drifted/corrupted weights

    handle = type("H", (), {"accessor": FakeCrossbar((4, 4))})()

    ok = apply_layer_reset(
        layer,
        [handle],
        original_weight,
        mapping_routine_for("DoubleColumn"),
        r_on=100.0,
        r_off=10000.0,
        scheme="DoubleColumn",
    )

    assert ok is True
    assert torch.equal(layer.weight.data, original_weight)
    assert torch.equal(handle.accessor.read(), original_weight)
