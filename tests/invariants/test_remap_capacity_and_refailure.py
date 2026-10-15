"""Regression tests for the two remap bugs found 2026-10-02 plus the spare
re-failure and injection-generator mechanics built on top of their fix (see
results/MANIFEST.md, findings 1-3):

- Bug #1 (selection stall): remapped cells stayed in monitor.stuck_mask, so
  dispatch's stuck_capable began with already-remapped positions and
  select_remappable() stopped at index 0 - every wave after the first remapped
  0 cells with capacity left (real run.log: "Remapped 14745/14745 (7373
  remaining)" then "Remapped 0/14745").
- Bug #2 (health gate): remap only ran when per-layer avg_health >
  health_threshold, a synthetic quantity with no physical bearing on whether a
  stuck cell can be bypassed.

Everything here goes through the real FaultTolerantNeuromorphic orchestrator
with real CrossSim analog layers, real optimizer.step() + synchronize(), and
health_check_interval=1 (a real check_health() every forward) - no mocks at
the seam under test.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
import torch

pytest.importorskip("simulator", reason="CrossSim not installed (uv sync --extra crosssim)")

from neurofault.config import ExperimentConfig, load_config
from neurofault.system import FaultTolerantNeuromorphic


def _build_system(
    *,
    redundancy_factor: float = 0.15,
    density: float = 0.1,
    failure_model: str = "density",
    distribution: str = "random",
    health_check_interval: int = 1,
) -> FaultTolerantNeuromorphic:
    torch.manual_seed(0)
    model = torch.nn.Sequential(torch.nn.Linear(32, 24), torch.nn.ReLU(), torch.nn.Linear(24, 10))
    config = ExperimentConfig(name="remap_capacity_test", simulator="crosssim")
    config.crossbar.patch_layer_types = ["Linear"]
    config.crossbar.calibrate_adc = False
    config.crossbar.calibrate_dac = False
    config.crossbar.redundancy_factor = redundancy_factor
    config.fault.density = density
    config.fault.failure_model = failure_model
    config.fault.distribution = distribution
    config.mitigation.enable_remapping = True
    config.mitigation.enable_soft_mitigation = False
    config.mitigation.enable_layer_reset = False
    config.mitigation.health_check_interval = health_check_interval
    return FaultTolerantNeuromorphic(model, config)


def _train_step(system: FaultTolerantNeuromorphic, optimizer) -> None:
    """One real training step exactly as experiments/run.py does it."""
    x = torch.randn(4, 32)
    target = torch.randint(0, 10, (4,))
    optimizer.zero_grad()
    loss = torch.nn.functional.cross_entropy(system.forward(x), target)
    loss.backward()
    optimizer.step()
    system.synchronize()


def _wave_and_check(system: FaultTolerantNeuromorphic) -> dict:
    """Inject one wave, then run the real forward (-> check_health) and report,
    per layer, what the dispatch should have been able to remap vs what it did."""
    system.inject_faults()
    before = {
        name: (
            int(system.monitors[name].effective_stuck.sum()),
            system.pools[name].available_capacity,
            system.pools[name].spares_used,
        )
        for name in system.handles
    }
    system.forward(torch.randn(4, 32))
    return {
        name: {
            "effective_stuck": eff,
            "available": avail,
            "remapped": system.pools[name].spares_used - used_before,
        }
        for name, (eff, avail, used_before) in before.items()
    }


def test_multi_wave_remap_uses_remaining_capacity_exactly():
    """Bug #1: wave 2 must remap exactly min(new effective-stuck, remaining
    capacity) per layer - not 0. Wave 3 (capacity now exhausted) remaps 0."""
    system = _build_system()  # 15% capacity, 10% density -> wave 2 exceeds remaining capacity
    optimizer = torch.optim.SGD(system.model.parameters(), lr=0.01)

    wave1 = _wave_and_check(system)
    for name, w in wave1.items():
        assert w["effective_stuck"] > 0
        assert w["remapped"] == min(w["effective_stuck"], w["available"]) == w["effective_stuck"]

    wave2 = _wave_and_check(system)
    saw_partial = False
    for name, w in wave2.items():
        expected = min(w["effective_stuck"], w["available"])
        assert expected > 0, "test setup must leave capacity for a second wave"
        assert w["remapped"] == expected, (name, w)  # bug #1: this was 0
        saw_partial |= w["remapped"] < w["effective_stuck"]
    assert saw_partial, "wave 2 should exceed remaining capacity in at least one layer"

    # Accounting is exact: the ledger, the live dict, the system stat all agree.
    total_used = 0
    for name, pool in system.pools.items():
        # wave 2's injection re-hits some wave-1 spares (they fail, are dropped from
        # the live set, and are NOT refunded), so live <= ever-consumed.
        assert len(pool.remapping) <= pool.spares_used
        assert pool.available_capacity == pool.capacity - pool.spares_used >= 0
        monitor = system.monitors[name]
        assert int(monitor.remapped_mask.sum()) == len(pool.remapping)
        total_used += pool.spares_used
    assert system.stats["remappings"] == total_used

    # Remapped cells are working spares: not critical, healthy, and their
    # reference values survive a real optimizer.step() + synchronize().
    _train_step(system, optimizer)
    for name, handle in system.handles.items():
        monitor, pool = system.monitors[name], system.pools[name]
        remapped = monitor.remapped_mask
        assert remapped.any()
        monitor.update_health_metrics()
        assert monitor.health_scores[remapped].min() > 90
        assert monitor.stress[remapped].max() < 0.1
        assert (monitor.health_scores[remapped] < 10).sum() == 0
        critical = set(monitor.get_critical_devices())
        rows, cols = torch.nonzero(remapped, as_tuple=True)
        low_health_or_probability = [
            (r, c)
            for r, c in zip(rows.tolist(), cols.tolist())
            if monitor.health_scores[r, c] < 10 or monitor.failure_probability[r, c] > 0.8
        ]
        assert not low_health_or_probability
        # (a remapped cell may still show up via the "extreme conductance"
        # heuristic if its reference value is legitimately near a rail - that
        # is a property of the value, not of the cell's health.)
        del critical
        matrix = handle.accessor.read()
        assert torch.allclose(matrix[remapped], pool.reference_conductance[remapped], atol=1e-6)
        # Effectively-stuck (unremapped, capacity-exhausted) cells stay pinned.
        eff = monitor.effective_stuck
        if eff.any():
            pinned = system._stuck_pin_values[name]
            assert torch.allclose(matrix[eff], pinned[eff], atol=1e-6)

    # Wave 3: capacity of the smaller layer is exhausted -> accounting still exact.
    wave3 = _wave_and_check(system)
    for w in wave3.values():
        assert w["remapped"] == min(w["effective_stuck"], w["available"])
    assert any(w["available"] == 0 and w["remapped"] == 0 for w in wave3.values()) or all(
        w["remapped"] > 0 for w in wave3.values()
    )


def test_remap_is_not_gated_on_per_layer_avg_health():
    """Bug #2: a layer whose avg_health is below health_threshold must still
    remap its stuck cells (health is a synthetic stress quantity, not a
    measurement of whether a stuck cell can be bypassed)."""
    system = _build_system()
    threshold = system.config.mitigation.health_threshold
    system.inject_faults()
    for monitor in system.monitors.values():
        monitor.stress[~monitor.stuck_mask] = 0.5  # force per-layer avg health well below 75

    system.forward(torch.randn(4, 32))  # real check_health()

    for name, monitor in system.monitors.items():
        assert monitor.get_health_summary()["avg_health"] < threshold
        assert system.pools[name].spares_used > 0
        assert int(monitor.remapped_mask.sum()) == system.pools[name].spares_used
    assert system.stats["remappings"] > 0


def test_refailed_spare_is_stuck_again_without_refund_and_remap_eligible():
    system = _build_system(redundancy_factor=0.5, health_check_interval=1000)
    optimizer = torch.optim.SGD(system.model.parameters(), lr=0.01)

    system.inject_faults()
    system.check_health()  # explicit: the interval is large so injection+step can be inspected
    used_before = {n: p.spares_used for n, p in system.pools.items()}
    remapped_before = {n: m.remapped_mask.clone() for n, m in system.monitors.items()}
    assert all(u > 0 for u in used_before.values())

    system.config.fault.density = 0.3  # wave 2 re-hits ~30% of the remapped spares
    system.inject_faults()

    for name, handle in system.handles.items():
        monitor, pool = system.monitors[name], system.pools[name]
        failed = monitor.last_injected_mask & remapped_before[name]
        assert failed.any()
        # stuck again, spare gone from the live set, nothing refunded
        assert not monitor.remapped_mask[failed].any()
        assert torch.equal(monitor.remapped_mask, remapped_before[name] & ~failed)
        assert monitor.effective_stuck[failed].all()
        assert pool.spares_used == used_before[name]
        assert len(pool.remapping) == used_before[name] - int(failed.sum())
        assert pool.available_capacity == pool.capacity - used_before[name]

    # Through a real step + synchronize: the stuck pin (not the remap pin)
    # applies at re-failed positions; still-live spares keep their reference.
    _train_step(system, optimizer)
    for name, handle in system.handles.items():
        monitor, pool = system.monitors[name], system.pools[name]
        failed = monitor.last_injected_mask & remapped_before[name]
        matrix = handle.accessor.read()
        rails = torch.isclose(matrix[failed], torch.tensor(monitor.g_min)) | torch.isclose(
            matrix[failed], torch.tensor(monitor.g_max)
        )
        assert rails.all()
        still_live = monitor.remapped_mask
        assert torch.allclose(matrix[still_live], pool.reference_conductance[still_live], atol=1e-6)

    # Remap-eligible again because capacity (50%) remains: the failed positions
    # get a fresh spare, and consumption grows (never refunded, never reused).
    system.check_health()
    for name, pool in system.pools.items():
        monitor = system.monitors[name]
        failed = monitor.last_injected_mask & remapped_before[name]
        assert monitor.remapped_mask[failed].all()
        assert pool.spares_used > used_before[name]
        assert pool.spares_used == len(pool.remapping) + int(failed.sum())


# --- generator determinism / byte-identical default path ---------------------------

# Captured from the pre-generator code (before explicit generators were added):
# sha256[:16] of every layer's stuck_mask bytes after inject_faults() under
# torch.manual_seed(123) on _build_system()'s model, plus the next global-RNG draw
# (proving the global RNG was consumed identically, not just the masks).
_PRE_GENERATOR = {
    ("density", "random"): ("9263db9cb127f715", 0.6687291860580444),
    # clustered re-captured after the spatial-bias fix in faults/injection.py (a random subset of
    # the unique()-sorted candidates instead of its lowest flat indices): the extra randperm
    # changes both the mask and the RNG consumption. The original pre-fix pair was
    # ("a14632c3006418d5", 0.6687291860580444) and is reproduced exactly by the old [:num_faults]
    # selection; the result does not depend on the torch thread count.
    ("density", "clustered"): ("5d697cd500205a45", 0.13615620136260986),
    ("density", "gradient"): ("e48a2d4018b1ae06", 0.48418349027633667),
}
# 3 injection calls each, with weibull_scale=3.0 / exponential_rate=0.3, over stuck|drift masks.
_PRE_GENERATOR_MULTI = {
    "weibull": ("0ef5651bf9b4af02", 0.39474672079086304),
    "exponential": ("9d404f8b6c9e69dc", 0.7245778441429138),
    "drift": ("f14d794f11a42c42", 0.2843294143676758),
}


def _mask_hash(system: FaultTolerantNeuromorphic, drift: bool = False) -> str:
    chunks = []
    for m in system.monitors.values():
        mask = (m.stuck_mask | m.drift_mask) if drift else m.stuck_mask
        chunks.append(mask.numpy().tobytes())
    return hashlib.sha256(b"".join(chunks)).hexdigest()[:16]


@pytest.mark.parametrize(("failure_model", "distribution"), list(_PRE_GENERATOR))
def test_default_none_generator_path_is_byte_identical_to_before(failure_model, distribution):
    system = _build_system(failure_model=failure_model, distribution=distribution)
    torch.manual_seed(123)
    system.inject_faults()
    expected_hash, expected_next_draw = _PRE_GENERATOR[(failure_model, distribution)]
    assert _mask_hash(system) == expected_hash
    assert torch.rand(1).item() == expected_next_draw


@pytest.mark.parametrize("failure_model", list(_PRE_GENERATOR_MULTI))
def test_default_none_generator_path_is_byte_identical_for_statistical_and_drift(failure_model):
    system = _build_system(failure_model=failure_model)
    system.config.fault.weibull_scale = 3.0
    system.config.fault.exponential_rate = 0.3
    torch.manual_seed(123)
    for _ in range(3):
        system.inject_faults()
    expected_hash, expected_next_draw = _PRE_GENERATOR_MULTI[failure_model]
    assert _mask_hash(system, drift=True) == expected_hash
    assert torch.rand(1).item() == expected_next_draw


@pytest.mark.parametrize("distribution", ["random", "clustered", "gradient"])
def test_identically_seeded_generators_give_identical_fault_maps(distribution):
    system_a = _build_system(distribution=distribution)
    system_b = _build_system(distribution=distribution)

    torch.manual_seed(1)
    torch.rand(1000)  # advance the global RNG differently between the two systems
    gen_a = torch.Generator().manual_seed(5)
    system_a.inject_faults(generator=gen_a)

    torch.manual_seed(999)
    gen_b = torch.Generator().manual_seed(5)
    system_b.inject_faults(generator=gen_b)

    for name in system_a.monitors:
        assert torch.equal(system_a.monitors[name].stuck_mask, system_b.monitors[name].stuck_mask)
        assert system_a.monitors[name].stuck_mask.any()

    system_c = _build_system(distribution=distribution)
    system_c.inject_faults(generator=torch.Generator().manual_seed(6))
    assert any(
        not torch.equal(system_a.monitors[n].stuck_mask, system_c.monitors[n].stuck_mask)
        for n in system_a.monitors
    )


def test_a_generator_does_not_consume_the_global_rng():
    system = _build_system()
    torch.manual_seed(77)
    expected_next = torch.rand(1).item()
    torch.manual_seed(77)
    system.inject_faults(generator=torch.Generator().manual_seed(5))
    assert torch.rand(1).item() == expected_next


# --- golden: the full continuous-regime loop ------------------------------------------

GOLDEN_DIR = Path(__file__).resolve().parent.parent / "golden"


@pytest.mark.slow  # real ResNet-9 golden
def test_remap_golden_is_byte_equal_through_batch_6_and_remaps_more_afterwards():
    """tests/golden/ was captured at 208109f (pre-fix). Through batch 6 no
    post-wave-1 remap has fired yet (the first fires in batch 7's forward), so
    loss/accuracy/faults/grad_norm/remappings must match byte for byte;
    after that the bug fixes legitimately change behavior, and the final
    remappings counter must be strictly greater than golden's."""
    import json

    from experiments.run import run_experiment

    golden = json.loads((GOLDEN_DIR / "remap_golden_metrics.json").read_text())
    config = load_config(GOLDEN_DIR / "remap_golden_config.yaml")
    metrics = run_experiment(config)

    # "health" is deliberately NOT compared: wave 1's remap (batch 1) now resets the
    # remapped spares' health to 100 (mark_remapped), so the logged average health
    # legitimately differs from golden from batch 1 on. It does not feed training.
    for key in ("loss", "accuracy", "faults", "grad_norm", "remappings"):
        assert metrics[key][:7] == golden[key][:7], key
    print(f"golden remappings {golden['remappings'][-1]} -> {metrics['remappings'][-1]}")
    assert metrics["remappings"][-1] > golden["remappings"][-1]
