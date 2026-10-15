"""system mechanics, through the real orchestrator (real CrossSim analog
layers, real optimizer.step() + synchronize(), no mocks at the seam):

- stuck-weight freeze: .weight at effectively-stuck cells stays constant across real
  SGD steps (momentum + weight decay) while other cells still train;
- the freeze is behavior-neutral for the continuous golden except a measured ulp-level
  grad_norm coupling (CrossSim re-derives array limits from the whole .weight matrix);
- reference_mode frozen vs living after a remap;
- remap ordering (random/magnitude/saliency/prefix) and the per-layer budget cap.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest
import torch

pytest.importorskip("simulator", reason="CrossSim not installed (uv sync --extra crosssim)")

from neurofault.config import ExperimentConfig, load_config
from neurofault.retraining.saliency import FisherTracker
from neurofault.system import FaultTolerantNeuromorphic

GOLDEN_DIR = Path(__file__).resolve().parent.parent / "golden"


def _build(
    *,
    freeze: bool | None = True,
    reference_mode: str = "frozen",
    redundancy_factor: float = 0.4,
    density: float = 0.1,
    seed: int = 5,
) -> FaultTolerantNeuromorphic:
    torch.manual_seed(0)
    model = torch.nn.Sequential(torch.nn.Linear(32, 24), torch.nn.ReLU(), torch.nn.Linear(24, 10))
    config = ExperimentConfig(name="ws3_test", simulator="crosssim", seed=seed)
    config.crossbar.patch_layer_types = ["Linear"]
    config.crossbar.calibrate_adc = False
    config.crossbar.calibrate_dac = False
    config.crossbar.redundancy_factor = redundancy_factor
    config.fault.density = density
    config.mitigation.enable_remapping = True
    config.mitigation.enable_soft_mitigation = False
    config.mitigation.enable_layer_reset = False
    config.mitigation.health_check_interval = 10**9  # remap only via remap_now()
    config.retraining.freeze_stuck_weights = freeze
    config.retraining.reference_mode = reference_mode
    return FaultTolerantNeuromorphic(model, config)


def _wm(handle) -> torch.Tensor:
    """The layer's .weight as a matrix view - independent of system internals (the
    system's own accessor returns None when the freeze is off)."""
    return handle.layer.weight.data.view(handle.accessor.shape[0], -1)


def _wave(system, seed: int = 1) -> dict[str, torch.Tensor]:
    """Inject one wave with a dedicated generator; return pre-injection .weight matrices."""
    before = {n: _wm(h).clone() for n, h in system.handles.items()}
    g = torch.Generator()
    g.manual_seed(seed)
    system.inject_faults(generator=g)
    system.apply_pins()
    return before


def _step(system, optimizer, gen: torch.Generator) -> None:
    x = torch.randn(8, 32, generator=gen)
    t = torch.randint(0, 10, (8,), generator=gen)
    optimizer.zero_grad()
    loss = torch.nn.functional.cross_entropy(system.model(x), t)
    loss.backward()
    optimizer.step()
    system.synchronize()


def _optimizer(system):
    return torch.optim.SGD(
        system.model.parameters(), lr=0.1, momentum=0.9, nesterov=True, weight_decay=1e-2
    )


@pytest.mark.parametrize("freeze", [True, False])
def test_stuck_weights_are_frozen_across_real_steps_and_others_still_train(freeze):
    system = _build(freeze=freeze)
    before = _wave(system)
    opt, gen = _optimizer(system), torch.Generator().manual_seed(3)
    for _ in range(5):
        _step(system, opt, gen)

    for name, handle in system.handles.items():
        stuck = system.monitors[name].effective_stuck
        assert stuck.any()
        w = _wm(handle)
        if freeze:
            # exactly the value-at-fault: no grad/momentum/weight-decay drift
            assert torch.equal(w[stuck], before[name][stuck])
        else:
            # sensitivity control: without the freeze the stuck weights DO drift
            assert not torch.equal(w[stuck], before[name][stuck])
        assert not torch.equal(w[~stuck], before[name][~stuck])  # live cells still train


def test_snapshot_is_never_overwritten_by_a_later_wave():
    system = _build()
    before = _wave(system, seed=1)
    opt, gen = _optimizer(system), torch.Generator().manual_seed(3)
    for _ in range(3):
        _step(system, opt, gen)
    first_stuck = {n: m.effective_stuck.clone() for n, m in system.monitors.items()}
    _wave(system, seed=2)  # second wave: new cells stick, old snapshots must survive
    for name, handle in system.handles.items():
        w = _wm(handle)
        assert torch.equal(w[first_stuck[name]], before[name][first_stuck[name]])


def test_freeze_off_is_the_default_for_continuous_regime():
    assert _build(freeze=None)._freeze_enabled is False


@pytest.mark.slow  # real ResNet-9 golden
def test_freeze_only_moves_grad_norm_ulps_on_the_continuous_golden():
    """The freeze is not strictly byte-neutral: a drifting stuck .weight that leaves the
    matrix's [min, max] makes CrossSim rescale every other cell by ~1e-8 at the next
    synchronize(). Everything except grad_norm must still match the golden exactly, and
    grad_norm to 1e-4 relative (measured max ~3e-6: the ulp noise is
    carried forward by training, not amplified) - i.e. the freeze changes no training outcome."""
    from experiments.run import run_experiment

    golden = json.loads((GOLDEN_DIR / "refactor_golden_metrics.json").read_text())
    config = load_config(GOLDEN_DIR / "refactor_golden_config.yaml")
    config = replace(config, retraining=replace(config.retraining, freeze_stuck_weights=True))
    metrics = run_experiment(config)
    metrics.pop("layer_health")
    metrics.pop("epoch_seconds")
    for key in golden:
        if key == "grad_norm":
            continue
        assert metrics[key] == golden[key], key
    for a, b in zip(metrics["grad_norm"], golden["grad_norm"]):
        assert a == pytest.approx(b, rel=1e-4)


def test_living_spare_trains_from_value_at_fault_frozen_spare_stays_pinned():
    results = {}
    for mode in ("frozen", "living"):
        system = _build(reference_mode=mode)
        before = _wave(system)
        remapped = system.remap_now(order="prefix")
        assert all(n > 0 for n in remapped.values())
        spare = {n: p.remapping.keys() for n, p in system.pools.items()}
        masks = {}
        for name, handle in system.handles.items():
            m = torch.zeros_like(system.monitors[name].stuck_mask)
            rows, cols = zip(*spare[name])
            m[list(rows), list(cols)] = True
            masks[name] = m
            assert torch.equal(system.monitors[name].remapped_mask, m)
        if mode == "living":
            for name, handle in system.handles.items():
                m = masks[name]
                # right after remap: spare holds the weight the cell had at fault, in BOTH
                # the matrix and .weight
                assert torch.equal(_wm(handle)[m], before[name][m])
                assert torch.allclose(handle.accessor.read()[m], before[name][m], atol=1e-6)
        opt, gen = _optimizer(system), torch.Generator().manual_seed(3)
        for _ in range(4):
            _step(system, opt, gen)
        results[mode] = (system, before, masks)

    system, before, masks = results["frozen"]
    for name, handle in system.handles.items():
        m = masks[name]
        ref = system.pools[name].reference_conductance
        assert torch.allclose(handle.accessor.read()[m], ref[m], atol=1e-6)  # pinned to reference
    system, before, masks = results["living"]
    for name, handle in system.handles.items():
        m = masks[name]
        w = _wm(handle)
        assert not torch.equal(w[m], before[name][m])  # the spare trained
        assert torch.allclose(handle.accessor.read()[m], w[m], atol=1e-6)  # matrix == weight


def test_living_orig_spare_starts_at_reference_then_trains():
    system = _build(reference_mode="living_orig")
    _wave(system)
    remapped = system.remap_now(order="prefix")
    assert all(n > 0 for n in remapped.values())
    masks = {}
    for name, handle in system.handles.items():
        m = torch.zeros_like(system.monitors[name].stuck_mask)
        rows, cols = zip(*system.pools[name].remapping.keys())
        m[list(rows), list(cols)] = True
        masks[name] = m
        ref = system.pools[name].reference_conductance
        # right after remap: spare holds the ORIGINAL reference, in BOTH the matrix and .weight
        assert torch.allclose(handle.accessor.read()[m], ref[m], atol=1e-6)
        assert torch.allclose(_wm(handle)[m], ref[m].to(_wm(handle).dtype), atol=1e-6)
    opt, gen = _optimizer(system), torch.Generator().manual_seed(3)
    for _ in range(4):
        _step(system, opt, gen)
    for name, handle in system.handles.items():
        m = masks[name]
        ref = system.pools[name].reference_conductance
        w = _wm(handle)
        assert not torch.allclose(w[m], ref[m].to(w.dtype), atol=1e-6)  # not pinned: it trained
        assert torch.allclose(handle.accessor.read()[m], w[m], atol=1e-6)  # matrix == weight


def test_living_requires_snapshots_and_per_cell_spares():
    with pytest.raises(ValueError, match="value-at-fault"):
        _build(freeze=False, reference_mode="living")


def _remapped_sets(system) -> dict[str, set]:
    return {n: set(p.remapping) for n, p in system.pools.items()}


def test_random_order_is_deterministic_seeded_and_off_the_global_rng():
    sets = []
    for _ in range(2):
        system = _build(seed=5)
        _wave(system)
        torch.manual_seed(123)
        state = torch.get_rng_state()
        system.remap_now(order="random", budget=6)
        assert torch.equal(torch.get_rng_state(), state)  # global RNG untouched
        sets.append(_remapped_sets(system))
    assert sets[0] == sets[1]

    system = _build(seed=5)
    _wave(system)
    system.remap_now(order="prefix", budget=6)
    assert _remapped_sets(system) != sets[0]  # random is not the row-major prefix

    other = _build(seed=6)
    _wave(other)
    other.remap_now(order="random", budget=6)
    assert _remapped_sets(other) != sets[0]


def _expected_top(system, before, score, k):
    out = {}
    for name, handle in system.handles.items():
        stuck = system.monitors[name].effective_stuck
        rows, cols = torch.nonzero(stuck, as_tuple=True)
        dev = handle.accessor.read()[rows, cols] - before[name][rows, cols]
        s = score(name, rows, cols, dev)
        top = torch.sort(s, descending=True, stable=True).indices[:k]
        out[name] = {(int(rows[i]), int(cols[i])) for i in top}
    return out


def test_magnitude_order_picks_largest_deviation_first_and_budget_caps_per_layer():
    system = _build()
    before = _wave(system)
    expected = _expected_top(system, before, lambda n, r, c, d: d.abs(), 4)
    remapped = system.remap_now(order="magnitude", budget=4)
    assert all(v == 4 for v in remapped.values())  # budget respected per layer
    assert _remapped_sets(system) == expected
    assert all(p.spares_used == 4 for p in system.pools.values())  # rest stays available
    # a second call with no cap spends more spares on what is left
    again = system.remap_now(order="magnitude")
    assert all(v > 0 for v in again.values())


def test_saliency_order_uses_fisher_times_squared_deviation():
    system = _build()
    tracker = FisherTracker.from_handles(system.handles)
    g = torch.Generator().manual_seed(9)
    for f in tracker.fisher.values():
        f.copy_(torch.rand(f.shape, generator=g))
    system.fisher = tracker
    before = _wave(system)

    def score(name, rows, cols, dev):
        fisher = tracker.fisher[name].reshape(system.handles[name].accessor.shape[0], -1)
        return fisher[rows, cols] * dev**2

    expected = _expected_top(system, before, score, 5)
    system.remap_now(order="saliency", budget=5)
    assert _remapped_sets(system) == expected


def test_saliency_without_a_tracker_is_a_clear_error():
    system = _build()
    _wave(system)
    with pytest.raises(ValueError, match="FisherTracker"):
        system.remap_now(order="saliency")


def test_remap_now_requires_remapping_enabled():
    system = _build()
    system.config.mitigation.enable_remapping = False
    with pytest.raises(ValueError, match="enable_remapping"):
        system.remap_now()


def test_remapped_spare_that_fails_again_gets_a_fresh_snapshot():
    system = _build(reference_mode="living", redundancy_factor=0.9, density=0.5)
    _wave(system, seed=1)
    system.remap_now(order="prefix")
    opt, gen = _optimizer(system), torch.Generator().manual_seed(3)
    for _ in range(3):
        _step(system, opt, gen)  # living spares drift away from their value-at-fault
    old = {n: s.clone() for n, s in system._stuck_weight_snapshot.items()}
    _wave(system, seed=2)  # density 0.5: many remapped spares are hit again
    refailed = 0
    for name, handle in system.handles.items():
        again = system.monitors[name].effective_stuck & system.monitors[name].last_injected_mask
        w = _wm(handle)
        sel = again & (system._stuck_weight_snapshot[name] != old[name])
        refailed += int(sel.sum())
        # fresh snapshot == the spare's CURRENT (trained) weight, which is now frozen
        assert torch.equal(w[sel], system._stuck_weight_snapshot[name][sel])
    assert refailed > 0
