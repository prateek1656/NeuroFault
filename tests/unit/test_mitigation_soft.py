from neurofault.crossbar.health_monitor import HealthMonitor
from neurofault.mitigation.soft import apply_soft_mitigation
from tests.unit.fakes import make_handle


def test_soft_mitigation_moves_toward_midpoint_only_on_targeted_cells():
    handle = make_handle(shape=(4, 4))
    monitor = HealthMonitor(handle)
    matrix = handle.accessor.read()
    matrix[0, 0] = monitor.g_min
    untouched_before = matrix[1, 1].item()
    handle.accessor.write(matrix)

    mitigated = apply_soft_mitigation(handle, monitor, [(0, 0)], blend=0.3)

    result = handle.accessor.read()
    assert mitigated == 1
    expected = monitor.g_min * 0.7 + ((monitor.g_min + monitor.g_max) / 2) * 0.3
    assert abs(result[0, 0].item() - expected) < 1e-9
    assert result[1, 1].item() == untouched_before  # untouched cell unchanged
