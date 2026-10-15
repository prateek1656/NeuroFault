import pytest
import torch

from neurofault.config import CrossbarConfig
from neurofault.crossbar.backends.crosssim_backend import mapping_routine_for
from neurofault.mitigation.reset import apply_layer_reset
from tests.unit.fakes import FakeCrossbar

pytest.importorskip("simulator", reason="CrossSim not installed (uv sync --extra crosssim)")

from neurofault.crossbar.backends.crosssim_backend import (
    _build_params,
    build_handles,
    patch_layers,
    synchronize,
)


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


def test_tile_shape_produces_expected_multi_core_partitioning():
    """Regression test for the CrossSim rows_max/cols_max axis inversion
    (verified during development, see the module docstring) - asserts the actual
    per-core shapes, not just that partitioning happened at all."""
    from simulator.algorithms.dnn.torch.linear import AnalogLinear

    model = torch.nn.Sequential(torch.nn.Linear(20, 16))  # weight shape (16, 20)
    crossbar_config = CrossbarConfig(patch_layer_types=["Linear"], tile_shape=(4, 20))
    device_params = {"r_on": 100.0, "r_off": 10000.0}

    patched = patch_layers(model, crossbar_config, None, device_params)
    lin = next(m for m in patched.modules() if isinstance(m, AnalogLinear))

    # tile_shape=(4, 20): rows constrained to 4 -> ceil(16/4)=4 cores, cols
    # unconstrained (20 fits in one chunk).
    assert lin.core.Ncores == 4
    assert lin.core.num_cores_row == 4
    assert lin.core.num_cores_col == 1

    # The logical accessor still sees the full, reassembled matrix.
    handles = build_handles(patched, device_params, crossbar_config)
    handle = next(iter(handles.values()))
    assert handle.accessor.shape == (16, 20)


def test_optimizer_step_alone_does_not_reach_the_analog_core():
    """Regression test for the real bug found during development: torch optimizers
    update parameters in-place, which CrossSim's own docs say does NOT
    auto-propagate to the analog core - without synchronize(), training is a
    no-op for the layer's actual forward-pass compute path."""
    model = torch.nn.Sequential(torch.nn.Linear(6, 4))
    crossbar_config = CrossbarConfig(patch_layer_types=["Linear"])
    device_params = {"r_on": 100.0, "r_off": 10000.0}
    patched = patch_layers(model, crossbar_config, None, device_params)
    lin = patched[0]

    matrix_before = lin.get_matrix().clone()
    optimizer = torch.optim.Adam(patched.parameters(), lr=0.1)
    x = torch.randn(4, 6)
    for _ in range(5):
        optimizer.zero_grad()
        patched(x).sum().backward()
        optimizer.step()

    assert torch.allclose(matrix_before, lin.get_matrix())  # unchanged without synchronize()


def test_synchronize_pushes_training_updates_into_the_analog_core():
    model = torch.nn.Sequential(torch.nn.Linear(6, 4))
    crossbar_config = CrossbarConfig(patch_layer_types=["Linear"])
    device_params = {"r_on": 100.0, "r_off": 10000.0}
    patched = patch_layers(model, crossbar_config, None, device_params)
    lin = patched[0]

    matrix_before = lin.get_matrix().clone()
    optimizer = torch.optim.Adam(patched.parameters(), lr=0.1)
    x = torch.randn(4, 6)
    for _ in range(5):
        optimizer.zero_grad()
        patched(x).sum().backward()
        optimizer.step()
        synchronize(patched)

    assert not torch.allclose(matrix_before, lin.get_matrix())


@pytest.mark.parametrize("topology", ["1T1R", "0T1R", "1S1R"])
def test_circuit_topology_constructs_and_fault_still_changes_output(topology):
    model = torch.nn.Sequential(torch.nn.Linear(6, 4))
    crossbar_config = CrossbarConfig(patch_layer_types=["Linear"], circuit_topology=topology)
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


def test_unknown_circuit_topology_raises():
    """CrossbarConfig.__post_init__ now fails this fast, at construction -
    _build_params()'s own check is unreachable via a validated config, but
    stays as defense-in-depth (see config.py)."""
    with pytest.raises(ValueError, match="Unknown crossbar_config.circuit_topology"):
        CrossbarConfig(patch_layer_types=["Linear"], circuit_topology="bogus")


def test_devices_per_synapse_uses_bit_sliced_core_and_still_trains():
    """Regression test for the real footgun found during development: weight_bits
    left at 0 alongside BITSLICED silently divides by zero inside CrossSim's
    own bitsliced_core.py - _build_params() must always set both together."""
    model = torch.nn.Sequential(torch.nn.Linear(6, 4))
    crossbar_config = CrossbarConfig(patch_layer_types=["Linear"], devices_per_synapse=4)
    device_params = {"r_on": 100.0, "r_off": 10000.0}

    patched = patch_layers(model, crossbar_config, None, device_params)
    handles = build_handles(patched, device_params, crossbar_config)
    handle = next(iter(handles.values()))

    assert handle.accessor.shape == (4, 6)  # logical shape unchanged by slicing

    x = torch.randn(2, 6)
    output_before = patched(x).detach().clone()
    matrix = handle.accessor.read().clone()
    matrix[0, 0] = matrix.max().item() * 100
    handle.accessor.write(matrix)
    assert not torch.allclose(output_before, patched(x).detach().clone())

    matrix_before_training = handle.accessor.read().clone()
    optimizer = torch.optim.Adam(patched.parameters(), lr=0.1)
    for _ in range(3):
        optimizer.zero_grad()
        patched(x).sum().backward()
        optimizer.step()
    synchronize(patched)

    assert not torch.allclose(matrix_before_training, handle.accessor.read())


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


def test_layer_reset_works_end_to_end_with_crosssim_mapping_routine():
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


def test_calibrate_dac_false_leaves_input_percentile_unset():
    """Default (False) must preserve every existing config/result exactly -
    see CrossbarConfig.calibrate_dac's docstring."""
    crossbar_config = CrossbarConfig(calibrate_dac=False)
    params = _build_params({"r_on": 100.0, "r_off": 10000.0}, crossbar_config)

    assert params.core.mapping.inputs.mvm.percentile is None
    assert params.core.mapping.inputs.vmm.percentile is None


def test_calibrate_dac_true_enables_input_percentile_scaling():
    crossbar_config = CrossbarConfig(calibrate_dac=True)
    params = _build_params({"r_on": 100.0, "r_off": 10000.0}, crossbar_config)

    assert params.core.mapping.inputs.mvm.percentile == 1.0
    assert params.core.mapping.inputs.vmm.percentile == 1.0


def test_calibrate_dac_prevents_real_input_clipping():
    """Real, direct reproduction of the bug: CrossSim's DAC
    defaults to a hardcoded [-1, 1] input clip range, independent of the
    real data flowing through it. Feed a layer input well outside that
    range and verify calibrate_dac=True recovers it - calibrate_dac=False
    must show real, measurable distortion from the same input.
    """
    torch.manual_seed(0)
    layer = torch.nn.Linear(16, 8, bias=False)
    device_params = {"r_on": 100.0, "r_off": 10000.0}
    # Real activations in this project's ResNet-9 reach ~5.6 - a scale like this, well outside
    # CrossSim's default [-1, 1] DAC
    # clip range, is what silently loses signal without the fix.
    x = torch.randn(4, 16) * 3.0
    with torch.no_grad():
        true_output = layer(x)

    def analog_output(calibrate_dac: bool) -> torch.Tensor:
        crossbar_config = CrossbarConfig(
            patch_layer_types=["Linear"],
            adc_resolution=16,
            dac_resolution=16,
            calibrate_dac=calibrate_dac,
        )
        patched = patch_layers(torch.nn.Sequential(layer), crossbar_config, None, device_params)
        with torch.no_grad():
            return patched(x)

    uncalibrated = analog_output(False)
    calibrated = analog_output(True)

    uncalibrated_cos = torch.nn.functional.cosine_similarity(
        uncalibrated.flatten(), true_output.flatten(), dim=0
    ).item()
    calibrated_cos = torch.nn.functional.cosine_similarity(
        calibrated.flatten(), true_output.flatten(), dim=0
    ).item()

    assert calibrated_cos > 0.999  # real fix recovers near-exact fidelity
    assert uncalibrated_cos < calibrated_cos - 0.01  # real, measurable gap without it
