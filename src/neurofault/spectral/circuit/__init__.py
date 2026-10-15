"""Circuit-level model of the analog eigenvector loop and the MVM baseline it must beat."""

from .loop import (
    LoopConfig,
    LoopModel,
    LoopTrace,
    analytic_rate,
    build_loop,
    eigen_reference,
    simulate,
    with_config,
)
from .mvm_timing import mvm_settle_detail, mvm_settle_time

__all__ = [
    "LoopConfig",
    "LoopModel",
    "LoopTrace",
    "analytic_rate",
    "build_loop",
    "eigen_reference",
    "mvm_settle_detail",
    "mvm_settle_time",
    "simulate",
    "with_config",
]
