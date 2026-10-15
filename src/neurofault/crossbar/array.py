"""Backend-agnostic crossbar abstraction.

CrossbarHandle is the single canonical reference to one memristive crossbar,
threaded through health monitoring, fault injection, and mitigation - never
a second, independently-constructed crossbar for the same layer (the structural fix for the original codebase's bugs #1 and #3).

This module itself imports no backend library (no crosssim or aihwkit) -
each backend's actual patching/accessor logic lives under
crossbar/backends/, imported lazily by whichever backend is selected via
ExperimentConfig.backend. That's what lets this module (and CrossbarHandle)
stay importable, and the rest of neurofault unit-testable, without any
particular backend installed - unit tests use a FakeCrossbar accessor.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch


class ConductanceAccessor(Protocol):
    """The minimal read/write interface every backend must provide for one
    crossbar's conductance state. Backends differ wildly in how they store
    this internally (get_matrix()/core.set_matrix() for CrossSim, get/
    set_weights() on a per-tile analog handle for AIHWKit) - everything in
    neurofault operates against this interface, never against a backend-
    specific attribute directly.
    """

    shape: tuple[int, int]

    def read(self) -> torch.Tensor: ...
    def write(self, matrix: torch.Tensor) -> None: ...


@dataclass
class CrossbarHandle:
    name: str
    accessor: ConductanceAccessor
    layer: torch.nn.Module
    device_params: dict
