"""A lightweight stand-in implementing ConductanceAccessor, exposing only
the interface the rest of neurofault actually depends on. Lets crossbar/
faults/mitigation logic be unit-tested without any real backend (crosssim/
aihwkit) installed.
"""

from __future__ import annotations

import torch

from neurofault.crossbar.array import CrossbarHandle


class FakeCrossbar:
    """Implements ConductanceAccessor directly - satisfies the Protocol
    structurally, no inheritance needed."""

    def __init__(self, shape: tuple[int, int], r_on: float = 100.0, r_off: float = 10000.0):
        self.shape = shape
        self.r_on = r_on
        self.r_off = r_off
        g_min, g_max = 1.0 / r_off, 1.0 / r_on
        self._matrix = torch.full(shape, (g_min + g_max) / 2)

    def read(self) -> torch.Tensor:
        return self._matrix

    def write(self, matrix: torch.Tensor) -> None:
        self._matrix = matrix.clone()


def make_handle(shape: tuple[int, int] = (8, 8), **crossbar_kwargs) -> CrossbarHandle:
    accessor = FakeCrossbar(shape, **crossbar_kwargs)
    layer = torch.nn.Linear(shape[1], shape[0])
    return CrossbarHandle(
        name="test_layer_crossbar_0",
        accessor=accessor,
        layer=layer,
        device_params={"r_on": accessor.r_on, "r_off": accessor.r_off},
    )
