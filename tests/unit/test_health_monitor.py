from neurofault.crossbar.health_monitor import HealthMonitor
from tests.unit.fakes import make_handle


def test_get_critical_devices_finds_extreme_conductance_cells():
    handle = make_handle(shape=(4, 4))
    monitor = HealthMonitor(handle)

    # Force two cells to the extremes; everything else stays mid-range and healthy.
    matrix = handle.accessor.read()
    matrix[0, 0] = monitor.g_min
    matrix[2, 3] = monitor.g_max
    handle.accessor.write(matrix)

    critical = set(monitor.get_critical_devices(extreme_threshold=0.02))

    assert (0, 0) in critical
    assert (2, 3) in critical
    # a mid-range, undisturbed cell must not be flagged
    assert (1, 1) not in critical


def test_get_critical_devices_uses_health_score_threshold():
    handle = make_handle(shape=(3, 3))
    monitor = HealthMonitor(handle)
    monitor.health_scores[1, 1] = 5.0  # below default health_threshold=10

    critical = monitor.get_critical_devices()

    assert (1, 1) in critical


def test_update_health_metrics_reduces_health_under_stress():
    handle = make_handle(shape=(2, 2))
    monitor = HealthMonitor(handle)
    initial_health = monitor.health_scores.clone()

    for _ in range(5):
        monitor.update_health_metrics(write_op=True)

    assert monitor.health_scores.mean().item() <= initial_health.mean().item()


def test_checksum_detection_flags_nothing_on_untouched_matrix():
    handle = make_handle(shape=(6, 6))
    monitor = HealthMonitor(handle)

    critical = monitor.get_critical_devices(detection_method="checksum")

    assert critical == []
    assert monitor.last_detection_cost == 6 + 6  # cheap path: rows + cols


def test_checksum_detection_falls_back_and_localizes_same_as_full_diff():
    handle = make_handle(shape=(6, 6))
    monitor = HealthMonitor(handle)
    matrix = handle.accessor.read()
    matrix[2, 4] = monitor.g_max
    handle.accessor.write(matrix)

    via_checksum = set(monitor.get_critical_devices(detection_method="checksum"))
    via_full_diff = set(monitor.get_critical_devices(detection_method="full_diff"))

    assert (2, 4) in via_checksum
    assert via_checksum == via_full_diff  # identical localization once a fault exists


def test_checksum_detection_cost_accumulates_only_the_cheap_path():
    handle = make_handle(shape=(6, 6))
    monitor = HealthMonitor(handle)

    for _ in range(3):
        monitor.get_critical_devices(detection_method="checksum")

    assert monitor.total_detection_cost == 3 * (6 + 6)  # never paid the rows*cols cost


def test_checksum_detection_misses_a_canceling_fault_pattern_full_diff_catches():
    """Real, known ABFT blind spot: faults arranged so both their row-sum
    AND column-sum contributions cancel exactly go undetected by checksum -
    documented as a limitation, not glossed over. full_diff still finds
    every one of these cells."""
    handle = make_handle(shape=(6, 6))
    monitor = HealthMonitor(handle)
    matrix = handle.accessor.read()

    # A 2x2 "conjugate" pattern: each row's and each column's sum contribution
    # cancels exactly (g_max-m) + (g_min-m) == 0 for every row/column touched.
    matrix[0, 0] = monitor.g_max
    matrix[0, 1] = monitor.g_min
    matrix[1, 0] = monitor.g_min
    matrix[1, 1] = monitor.g_max
    handle.accessor.write(matrix)

    via_checksum = monitor.get_critical_devices(detection_method="checksum")
    via_full_diff = set(monitor.get_critical_devices(detection_method="full_diff"))

    assert via_checksum == []  # the real blind spot
    assert {(0, 0), (0, 1), (1, 0), (1, 1)} <= via_full_diff
