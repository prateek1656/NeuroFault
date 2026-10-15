"""Soft mitigation: nudge a device's conductance partway toward mid-range.

Models correction of drift/variability, not a fix for a hardware-stuck
device (a real stuck-at-HRS/LRS device can't respond to a write pulse at
all - that distinction belongs to the caller/dispatch layer, which decides
whether soft mitigation is the appropriate strategy for a given device).
"""

from __future__ import annotations

import logging

from neurofault.crossbar.array import CrossbarHandle
from neurofault.crossbar.health_monitor import HealthMonitor

logger = logging.getLogger(__name__)


def apply_soft_mitigation(
    handle: CrossbarHandle,
    monitor: HealthMonitor,
    critical_devices: list[tuple[int, int]],
    blend: float = 0.3,
) -> int:
    """Move each targeted cell's conductance `blend` fraction toward g_mid.

    Writes through handle.accessor - the same interface every other module
    uses for this exact crossbar, regardless of backend.
    """
    if not critical_devices:
        return 0

    g_mid = (monitor.g_min + monitor.g_max) / 2
    matrix = handle.accessor.read()
    mitigated = 0
    for row, col in critical_devices:
        matrix[row, col] = matrix[row, col] * (1 - blend) + g_mid * blend
        mitigated += 1

    handle.accessor.write(matrix)
    logger.info("Soft-mitigated %d devices in %s", mitigated, handle.name)
    return mitigated
