import pytest
import torch

from neurofault.config import FaultConfig
from neurofault.crossbar.health_monitor import HealthMonitor
from neurofault.faults.injection import inject_faults
from tests.unit.fakes import make_handle


def test_custom_stuck_at_injects_requested_density():
    handle = make_handle(shape=(10, 10))  # 100 cells
    monitor = HealthMonitor(handle)
    config = FaultConfig(density=0.2)

    injected = inject_faults(handle, monitor, config)

    assert injected == 20
    matrix = handle.accessor.read()
    at_extreme = ((matrix == monitor.g_min) | (matrix == monitor.g_max)).sum().item()
    assert at_extreme == 20


def test_zero_density_injects_nothing():
    handle = make_handle(shape=(10, 10))
    monitor = HealthMonitor(handle)
    config = FaultConfig(density=0.0)

    assert inject_faults(handle, monitor, config) == 0


def test_clustered_distribution_stays_within_bounds():
    handle = make_handle(shape=(8, 8))
    monitor = HealthMonitor(handle)
    config = FaultConfig(density=0.1, distribution="clustered")

    injected = inject_faults(handle, monitor, config)

    assert injected > 0  # exact count varies with cluster overlap, just must be sane
    assert injected <= int(64 * 0.1) + 9  # generous bound: centers * up to 9 neighbors


def test_weibull_without_scale_raises_at_config_construction():
    with pytest.raises(ValueError, match="weibull_scale"):
        FaultConfig(failure_model="weibull")


def test_exponential_without_rate_raises_at_config_construction():
    with pytest.raises(ValueError, match="exponential_rate"):
        FaultConfig(failure_model="exponential")


def test_weibull_model_eventually_fails_most_cells_past_scale():
    """Monte Carlo check, not just 'did not raise': with weibull_scale well
    below the number of simulated cycles run, most cells should end up
    stuck - a real statistical behavior check."""
    torch.manual_seed(0)
    handle = make_handle(shape=(50, 50))  # 2500 cells
    monitor = HealthMonitor(handle)
    config = FaultConfig(failure_model="weibull", weibull_shape=2.0, weibull_scale=10.0)

    for _ in range(30):  # 30 >> scale=10, CDF(30) ~= 1 - e^-9 ~= 0.9999
        inject_faults(handle, monitor, config)

    assert monitor.stuck_mask.float().mean().item() > 0.9


def test_statistical_model_never_resamples_already_stuck_cells():
    torch.manual_seed(0)
    handle = make_handle(shape=(30, 30))
    monitor = HealthMonitor(handle)
    config = FaultConfig(failure_model="exponential", exponential_rate=0.05)

    inject_faults(handle, monitor, config)
    stuck_after_first = monitor.stuck_mask.clone()
    cycles_of_stuck = monitor.cycles[stuck_after_first].clone()

    inject_faults(handle, monitor, config)

    # Already-stuck cells must not have aged further (they were excluded
    # from the surviving-cells cycle increment).
    assert torch.equal(monitor.cycles[stuck_after_first], cycles_of_stuck)
    # Once stuck, always stuck - the mask only grows.
    assert torch.all(monitor.stuck_mask[stuck_after_first])


def test_drift_shifts_toward_hrs_without_marking_stuck():
    """Real failure mode found missing during development: soft_mitigation's own docstring says it corrects
    drift, not stuck-at faults, but this project never generated drift
    before now. Drifted cells must stay writable/functional (never
    stuck_mask) and shift toward g_min (the default "toward_hrs" bias)."""
    handle = make_handle(shape=(10, 10))
    monitor = HealthMonitor(handle)
    matrix_before = handle.accessor.read().clone()
    config = FaultConfig(density=0.3, failure_model="drift", drift_magnitude=0.1)

    n = inject_faults(handle, monitor, config)

    assert n == 30
    assert monitor.drift_mask.sum().item() == 30
    assert not monitor.stuck_mask.any()
    matrix_after = handle.accessor.read()
    shift = matrix_after[monitor.drift_mask] - matrix_before[monitor.drift_mask]
    assert torch.all(shift < 0)  # toward_hrs -> toward g_min
    assert torch.allclose(shift.abs(), torch.full_like(shift, 0.1 * monitor.g_range), atol=1e-5)


def test_drift_accumulates_and_clamps_at_the_rail():
    """Repeated drift events on the same cell must compound (real retention
    loss is cumulative), but never overshoot past g_min/g_max."""
    handle = make_handle(shape=(4, 4))
    monitor = HealthMonitor(handle)
    config = FaultConfig(density=1.0, failure_model="drift", drift_magnitude=0.3)

    for _ in range(10):  # 10 * 0.3 * g_range would wildly overshoot without clamping
        inject_faults(handle, monitor, config)

    matrix = handle.accessor.read()
    assert torch.allclose(matrix, torch.full_like(matrix, monitor.g_min), atol=1e-6)


def test_drift_random_per_cell_direction_is_decided_once_and_reused():
    """Real retention drift is monotonic per-device, not a coin flip every
    injection call - a cell's direction must stay fixed once assigned."""
    torch.manual_seed(0)
    handle = make_handle(shape=(20, 20))
    monitor = HealthMonitor(handle)
    config = FaultConfig(
        density=0.5, failure_model="drift", drift_direction="random_per_cell", drift_magnitude=0.02
    )

    inject_faults(handle, monitor, config)
    sign_after_first = monitor.drift_sign.clone()
    drifted_after_first = monitor.drift_mask.clone()

    inject_faults(handle, monitor, config)

    already_drifted = drifted_after_first & monitor.drift_mask
    assert already_drifted.any()
    assert torch.equal(sign_after_first[already_drifted], monitor.drift_sign[already_drifted])


def test_clustered_selection_is_not_biased_to_low_rows_and_random_unchanged():
    from neurofault.faults.injection import _select_fault_indices

    dev = torch.device("cpu")
    g = torch.Generator().manual_seed(0)
    idx = _select_fault_indices(256, 256, 6554, "clustered", dev, g)
    assert idx.numel() == 6554 and idx.unique().numel() == 6554
    mean_row = (idx // 256).float().mean().item()
    assert 110 < mean_row < 145  # was ~75 when the lowest flat indices were kept; ~127 expected
    # random branch byte-identical to the plain randperm prefix
    g1 = torch.Generator().manual_seed(3)
    g2 = torch.Generator().manual_seed(3)
    assert torch.equal(
        _select_fault_indices(64, 64, 100, "random", dev, g1),
        torch.randperm(64 * 64, device=dev, generator=g2)[:100],
    )
