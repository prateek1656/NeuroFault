"""Regression tests for three real bugs found during development while running the
first full track-1 comparison: soft_mitigation_only, remapping_only, and
layer_reset_only produced bit-identical results despite soft-mitigation and
remapping actually firing (34M/71.7M cell-touches logged) - verified to
reproduce against the pre-fix code (verified directly), not written to pass against
already-correct code:

- Bug A: mitigation writes (remap) only touch conductance via
  handle.accessor.write() - system.synchronize() (called after every
  optimizer.step()) re-derives conductance from .weight every batch, erasing
  the correction before any forward pass benefits from it. Verified directly:
  a 122-cell remap was overwritten by one real training step + synchronize().
- Bug B: layer_reset was dead code on both backends - system.py checked
  `hasattr(module, "crossbars")`, an attribute no patched layer class in
  either backend ever defines, so reset_fn was always None regardless of
  config or health.
- Bug C: stuck-at faults self-healed within one batch in every mode except
  fault_aware_retraining (covered separately in tests/unit/test_system_
  retraining.py, generalized there) - not duplicated here.

Uses FakeCrossbar/make_handle via a fake backend module (same pattern as
tests/unit/test_system_retraining.py) so these exercise the real
system.py/dispatch.py/remap.py/self_healing.py orchestration without needing
crosssim or aihwkit installed.
"""

from __future__ import annotations

import torch

from neurofault.config import ExperimentConfig
from neurofault.crossbar.array import CrossbarHandle
from neurofault.mitigation.dispatch import MitigationOutcome
from neurofault.models.cnn import build_model
from neurofault.system import FaultTolerantNeuromorphic
from tests.unit.fakes import FakeCrossbar


def _identity_mapping_routine(weight: torch.Tensor, r_on: float, r_off: float, scheme: str):
    """Mirrors crosssim_backend.py's/aihwkit_backend.py's own
    mapping_routine_for(): weight and conductance are the same values, just
    reshaped - real fake backends in production define this; the plain
    _FakeBackend in test_system_retraining.py doesn't need to, since it never
    exercises layer_reset."""
    rows = weight.shape[0]
    return weight.reshape(rows, -1)


def _system_with_fake_backend(monkeypatch, config: ExperimentConfig):
    """Unlike tests/unit/test_system_retraining.py's fixture, the handle's
    `.layer` here MUST be a real submodule of the model (fc, an
    nn.Linear(512, 10)) - not an independently-constructed one - because
    these tests exercise system.py's own layer-handle discovery
    (`{h.layer for h in self.handles.values()}` matched against
    `model.named_modules()`), which is exactly the Bug B regression these
    tests cover. An unrelated, disconnected `.layer` would never match,
    regardless of whether the discovery logic is correct or still broken."""
    import neurofault.system as system_module

    built = {}

    class _FakeBackend:
        @staticmethod
        def patch_layers(model, crossbar_config, device_cls, device_params, calibrated_ranges=None):
            return model

        @staticmethod
        def build_handles(model, device_params, crossbar_config):
            layer = model.fc
            accessor = FakeCrossbar(tuple(layer.weight.shape))
            handle = CrossbarHandle(
                name="fc_crossbar_0",
                accessor=accessor,
                layer=layer,
                device_params={"r_on": accessor.r_on, "r_off": accessor.r_off},
            )
            built["handle"] = handle
            return {handle.name: handle}

        @staticmethod
        def synchronize(model):
            # Mirrors what CrossSim's real synchronize() does (see
            # crosssim_backend.py): blindly re-derive conductance from
            # .weight, oblivious to any pin. A no-op here would make these
            # tests pass trivially regardless of the fix, since nothing
            # would ever erase a pin in the first place.
            handle = built["handle"]
            handle.accessor.write(model.fc.weight.data.reshape(handle.accessor.shape))

        @staticmethod
        def mapping_routine_for(scheme):
            return _identity_mapping_routine

    monkeypatch.setattr(system_module, "_backend_module", lambda simulator: _FakeBackend)
    system = FaultTolerantNeuromorphic(build_model(config), config)
    return system, built["handle"]


def test_remap_survives_a_real_optimizer_step_and_synchronize(monkeypatch):
    """Bug A regression: a remap must still be in place after the training
    loop's real optimizer.step() + system.synchronize(), not just at the
    instant apply_remap() writes it."""
    from neurofault.mitigation.remap import apply_remap

    config = ExperimentConfig(model_arch="tiny_cnn", name="test_remap_persistence")
    system, handle = _system_with_fake_backend(monkeypatch, config)
    pool = next(iter(system.pools.values()))

    reference_value = pool.reference_conductance[0, 0].item()
    matrix = handle.accessor.read()
    matrix[0, 0] = 999.0  # simulate a fault
    handle.accessor.write(matrix)

    remapped = apply_remap(handle, pool, [(0, 0)])
    assert remapped == 1
    assert handle.accessor.read()[0, 0].item() == reference_value

    optimizer = torch.optim.Adam(system.model.parameters(), lr=0.1)
    x = torch.randn(2, 3, 32, 32)
    targets = torch.randint(0, 10, (2,))
    criterion = torch.nn.CrossEntropyLoss()

    optimizer.zero_grad()
    loss = criterion(system.forward(x), targets)
    loss.backward()
    optimizer.step()
    system.synchronize()

    assert handle.accessor.read()[0, 0].item() == reference_value


def test_many_remaps_survive_a_real_optimizer_step_and_synchronize(monkeypatch):
    """Regression for the second, separate perf bug found 2026-09-30 on a real GPU
    campaign run: system.py's
    _reapply_pins() re-pins the FULL accumulated pool.remapping set on every
    synchronize() call (i.e. every batch), not just newly-remapped cells - the
    single-remap test above wouldn't catch a bug in the fancy-indexed multi-position
    form (system.py's fix), since a single-element case can behave differently from
    a batch one. This exercises several simultaneously-remapped positions to confirm
    the vectorized write still pins all of them correctly, not just the first/last."""
    config = ExperimentConfig(model_arch="tiny_cnn", name="test_many_remaps_persistence")
    system, handle = _system_with_fake_backend(monkeypatch, config)
    pool = next(iter(system.pools.values()))

    from neurofault.mitigation.remap import apply_remap

    positions = [(0, 0), (1, 1), (2, 2), (0, 2), (2, 0)]
    reference_values = {pos: pool.reference_conductance[pos].item() for pos in positions}
    matrix = handle.accessor.read()
    for pos in positions:
        matrix[pos] = 999.0  # simulate faults
    handle.accessor.write(matrix)

    remapped = apply_remap(handle, pool, positions)
    assert remapped == len(positions)
    for pos in positions:
        assert handle.accessor.read()[pos].item() == reference_values[pos]

    optimizer = torch.optim.Adam(system.model.parameters(), lr=0.1)
    x = torch.randn(2, 3, 32, 32)
    targets = torch.randint(0, 10, (2,))
    criterion = torch.nn.CrossEntropyLoss()

    optimizer.zero_grad()
    loss = criterion(system.forward(x), targets)
    loss.backward()
    optimizer.step()
    system.synchronize()

    matrix_after = handle.accessor.read()
    for pos in positions:
        assert matrix_after[pos].item() == reference_values[pos], (
            f"{pos} was not re-pinned correctly by the vectorized _reapply_pins"
        )


def test_row_col_remap_survives_a_real_optimizer_step_and_synchronize(monkeypatch):
    """Same as above for the row_col_granularity scheme - _reapply_pins()
    branches on pool type and must cover both."""
    from neurofault.crossbar.self_healing import RowColumnRedundancyPool
    from neurofault.mitigation.remap import apply_remap

    config = ExperimentConfig(model_arch="tiny_cnn", name="test_row_col_remap_persistence")
    config.crossbar.redundancy_scheme = "row_col_granularity"
    system, handle = _system_with_fake_backend(monkeypatch, config)
    pool = next(iter(system.pools.values()))
    assert isinstance(pool, RowColumnRedundancyPool)

    reference_row = pool.reference_conductance[0, :].clone()
    matrix = handle.accessor.read()
    matrix[0, :] = 999.0
    handle.accessor.write(matrix)

    critical = [(0, c) for c in range(handle.accessor.shape[1])]  # whole row -> condemns it
    remapped = apply_remap(handle, pool, critical)
    assert remapped == 1
    assert torch.equal(handle.accessor.read()[0, :], reference_row)

    optimizer = torch.optim.Adam(system.model.parameters(), lr=0.1)
    x = torch.randn(2, 3, 32, 32)
    targets = torch.randint(0, 10, (2,))
    criterion = torch.nn.CrossEntropyLoss()

    optimizer.zero_grad()
    loss = criterion(system.forward(x), targets)
    loss.backward()
    optimizer.step()
    system.synchronize()

    assert torch.equal(handle.accessor.read()[0, :], reference_row)


def test_reset_fn_is_constructed_for_a_real_patched_layer(monkeypatch):
    """Bug B regression: _layer_handles/_original_weights must actually be
    populated for a patched layer (they used to always be empty, on both
    backends, because of the hasattr(module, "crossbars") check)."""
    config = ExperimentConfig(model_arch="tiny_cnn", name="test_reset_wiring")
    system, handle = _system_with_fake_backend(monkeypatch, config)

    assert system._layer_handles, "no patched layer was discovered for layer_reset"
    assert system._original_weights, "no original weights were captured for layer_reset"
    assert system._find_layer_name(handle) is not None


def test_layer_reset_fires_through_real_check_health_under_low_health(monkeypatch):
    """Bug B regression, end to end: with enable_layer_reset=True and health
    forced into the reset-eligible range, system.check_health() must
    actually construct and invoke reset_fn - not silently skip it."""
    config = ExperimentConfig(model_arch="tiny_cnn", name="test_layer_reset_fires")
    config.mitigation.enable_layer_reset = True
    config.mitigation.health_threshold = 75.0
    system, handle = _system_with_fake_backend(monkeypatch, config)
    monitor = next(iter(system.monitors.values()))

    # Force a low-health, critical-device state directly (isolating dispatch
    # wiring from HealthMonitor's own decay dynamics, already covered
    # elsewhere).
    monitor.stuck_mask[:] = True
    monitor.health_scores[:] = 0.0
    monitor.stability_index[:] = 0.1

    layer = handle.layer
    original_weight = layer.weight.data.clone()
    layer.weight.data.fill_(999.0)  # simulate drifted training weights

    outcomes = system.check_health()
    outcome = next(iter(outcomes.values()))

    assert "layer_reset" in outcome.actions_taken
    assert torch.equal(layer.weight.data, original_weight)


def test_layer_reset_only_touches_critical_cells_not_the_whole_layer(monkeypatch):
    """The actual fix, not just the wiring: real design flaw found and fixed
    2026-09-13 (see reset.py's module docstring) - whole-layer reset
    discarded every batch of legitimate learning along with the fault
    correction, measured robustly worse than no mitigation across 3 seeds
    (11.80% vs 41.57% mean final accuracy). Only the specific critical cell
    should be restored to its pre-training value; every other cell's
    trained value must survive untouched."""
    config = ExperimentConfig(model_arch="tiny_cnn", name="test_targeted_reset")
    config.mitigation.enable_layer_reset = True
    config.mitigation.health_threshold = 75.0
    system, handle = _system_with_fake_backend(monkeypatch, config)
    monitor = next(iter(system.monitors.values()))

    # Mark only rows 0-2 (of 10) stuck: update_health_metrics() snapshots
    # and restores stuck cells' health_scores exactly (see health_monitor.py
    # - this is the OTHER, already-fixed self-healing bug), so this reliably
    # pulls avg_health below the 75.0 threshold (~30% of cells at 0 health
    # -> ~70.0 average) without needing many real training steps - the rest
    # of the layer is untouched and should survive the reset.
    monitor.stuck_mask[0:3, :] = True
    monitor.health_scores[0:3, :] = 0.0
    monitor.stability_index[:] = 1.0

    layer = handle.layer
    original_weight = layer.weight.data.clone()
    trained_weight = torch.randn_like(layer.weight.data)
    layer.weight.data.copy_(trained_weight)

    outcomes = system.check_health()
    outcome = next(iter(outcomes.values()))
    assert outcome.layer_reset is True

    rows = layer.weight.data.shape[0]
    reset_view = layer.weight.data.reshape(rows, -1)
    original_view = original_weight.reshape(rows, -1)
    trained_view = trained_weight.reshape(rows, -1)

    # The stuck rows are restored to their pre-training values...
    assert torch.equal(reset_view[0:3, :], original_view[0:3, :])
    # ...but everywhere else must still hold the TRAINED values, not the
    # original - this is exactly the property a whole-layer reset violates.
    assert torch.equal(reset_view[3:, :], trained_view[3:, :])


def test_layer_reset_never_fires_on_heuristically_critical_non_stuck_cells(monkeypatch):
    """Real precision fix found during development: layer_reset used to target the raw get_critical_devices()
    heuristic set, which can flag cells that are merely legitimately-trained
    near an extreme value, not actually faulty. A mixed scenario - some
    cells genuinely stuck (driving avg_health below threshold, same
    mechanism as the sibling test above), one merely heuristically extreme
    but never stuck - must reset only the verified-stuck cells."""
    config = ExperimentConfig(model_arch="tiny_cnn", name="test_reset_precision")
    config.mitigation.enable_layer_reset = True
    config.mitigation.health_threshold = 75.0
    system, handle = _system_with_fake_backend(monkeypatch, config)
    monitor = next(iter(system.monitors.values()))

    # Rows 0-2 genuinely stuck - drives avg_health below threshold (update_
    # health_metrics() would otherwise recompute health fresh from live
    # conductance every call and never let it stay low without this).
    monitor.stuck_mask[0:3, :] = True
    monitor.health_scores[0:3, :] = 0.0
    monitor.stability_index[:] = 1.0
    # Row 3: never stuck, but its conductance happens to sit at an extreme
    # value (e.g. a legitimately-trained weight) - heuristically flagged
    # critical too, must never be touched.
    matrix = handle.accessor.read()
    matrix[3, :] = monitor.g_max
    handle.accessor.write(matrix)

    layer = handle.layer
    original_weight = layer.weight.data.clone()
    trained_weight = torch.randn_like(layer.weight.data)
    layer.weight.data.copy_(trained_weight)

    outcomes = system.check_health()
    outcome = next(iter(outcomes.values()))
    assert outcome.layer_reset is True  # fires because of the genuinely stuck rows

    rows = layer.weight.data.shape[0]
    reset_view = layer.weight.data.reshape(rows, -1)
    original_view = original_weight.reshape(rows, -1)
    trained_view = trained_weight.reshape(rows, -1)

    assert torch.equal(reset_view[0:3, :], original_view[0:3, :])  # stuck rows reset
    assert torch.equal(reset_view[3, :], trained_view[3, :])  # heuristic-only row left alone
    assert torch.equal(reset_view[4:, :], trained_view[4:, :])  # everything else untouched


def test_layer_reset_is_throttled_by_cooldown_not_constant(monkeypatch):
    """Real design flaw found and fixed during development: with no cooldown,
    layer_reset fired on essentially every check_health() call once health
    dropped below threshold (measured: 222 times over 1257 batches in a real
    run) - wiping the whole layer back to random init far more often than a
    'last resort' mitigation should, actively hurting accuracy. With
    layer_reset_cooldown_checks set, a second immediate check_health() call
    (health still low, same layer) must NOT reset again; only after the
    cooldown elapses should it be eligible again."""
    config = ExperimentConfig(model_arch="tiny_cnn", name="test_layer_reset_cooldown")
    config.mitigation.enable_layer_reset = True
    config.mitigation.health_threshold = 75.0
    config.mitigation.layer_reset_cooldown_checks = 3
    system, handle = _system_with_fake_backend(monkeypatch, config)
    monitor = next(iter(system.monitors.values()))

    monitor.stuck_mask[:] = True
    monitor.health_scores[:] = 0.0
    monitor.stability_index[:] = 0.1

    fired = []
    for _ in range(5):
        handle.layer.weight.data.fill_(999.0)  # simulate drift each round
        outcomes = system.check_health()
        outcome = next(iter(outcomes.values()))
        # outcome.layer_reset is whether the reset actually happened -
        # actions_taken records every *attempt* regardless of cooldown
        # (dispatch.py appends the label unconditionally once reset_fn is
        # called), so it can't distinguish a throttled attempt from a real
        # reset.
        fired.append(outcome.layer_reset)

    # Fires on the first call (no prior reset), then withheld for
    # cooldown_checks calls, then eligible again.
    assert fired == [True, False, False, True, False]


def test_soft_mitigation_survives_one_synchronize_then_releases(monkeypatch):
    """Soft-mitigation's correction must survive exactly the next
    synchronize() (Bug A's fix), then no longer be pinned afterward - it
    models drift correction, not a permanent freeze, so ordinary training
    should resume shaping the cell from the corrected baseline."""
    config = ExperimentConfig(model_arch="tiny_cnn", name="test_soft_one_shot")
    system, handle = _system_with_fake_backend(monkeypatch, config)

    matrix = handle.accessor.read()
    matrix[1, 1] = 42.0
    handle.accessor.write(matrix)

    outcome = MitigationOutcome(soft_mitigated=1, soft_mitigated_positions=[(1, 1)])
    system._pin_soft_mitigation(handle.name, handle, outcome)

    # First synchronize: the pin must survive an unrelated full overwrite,
    # exactly like the real training loop's optimizer.step() would cause.
    handle.accessor.write(torch.randn(handle.accessor.shape))
    system.synchronize()
    assert handle.accessor.read()[1, 1].item() == 42.0

    # Second synchronize, after a real training step and no new mitigation in
    # between: the pin must be released - the cell should follow ordinary
    # training instead of staying frozen at 42.0 forever.
    optimizer = torch.optim.Adam(system.model.parameters(), lr=0.5)
    x = torch.randn(2, 3, 32, 32)
    targets = torch.randint(0, 10, (2,))
    criterion = torch.nn.CrossEntropyLoss()
    optimizer.zero_grad()
    loss = criterion(system.forward(x), targets)
    loss.backward()
    optimizer.step()
    system.synchronize()

    assert handle.accessor.read()[1, 1].item() != 42.0
