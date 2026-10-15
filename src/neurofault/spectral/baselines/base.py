"""Shared plumbing for the B0-B9 baseline ladder.

WHY this module exists: every baseline must (1) touch the tile only through the counted
TileReaderLike, (2) report the cost of exactly its own reads, and (3) emit a ``W_eff`` so all
methods are scored by one metric. Step (3) is the awkward one: a gain-type method learns
multiplicative factors from reads but the effective operator ``diag(g) W(t) diag(h)`` has to be
materialised from W(t), which the method must NOT use for estimation. We therefore give methods a
``bind(w_now, w0)`` hook used ONLY to materialise W_eff (a linear digital post-scale of the
analog output is exactly this operator) or, for the oracle, to hand back the truth. Estimators
read ``reader`` exclusively; the harness binds the providers. The fitted gains are also in
``Correction.meta`` so a harness can apply them itself.

Axis convention: W is (m outputs, n inputs). A *row* gain scales output i (the digital
post-multiplier on the ADC result); a *column* gain scales input j. AIHWKit's
``PerColumnDriftCompensation`` reads ``eye(n)`` into an (n, m) tensor and averages dim=0, i.e. it
yields one factor per OUTPUT = one factor per W row. Our ``PerRow`` is therefore the AIHWKit
"per column" method; ``PerColumn`` here is the per-INPUT gain (needs backward reads).
"""

from __future__ import annotations

from collections.abc import Callable

import torch

from neurofault.spectral.types import (
    DTYPE,
    Compensator,
    Correction,
    Detector,
    FaultEstimate,
    Localizer,
    ReadCost,
    TileReaderLike,
)

__all__ = [
    "Bindable",
    "Compensator",
    "Correction",
    "Detector",
    "FaultEstimate",
    "Localizer",
    "ReadCost",
    "TileReaderLike",
    "apply_gains",
    "bind_all",
    "ls_gain",
    "make_probes",
]


def make_probes(kind: str, n: int, p: int, gen: torch.Generator | None = None) -> torch.Tensor:
    """(n, P) probe matrix. kinds: ones | rademacher | hadamard | identity | onehot.

    identity = all n unit vectors (p ignored); onehot = p distinct random unit vectors;
    hadamard = first p columns of a Sylvester Hadamard matrix cropped to n rows (orthogonal
    only when n is a power of two)."""
    if kind == "ones":
        return torch.ones(n, 1, dtype=DTYPE)
    if kind == "identity":
        return torch.eye(n, dtype=DTYPE)
    if kind == "onehot":
        idx = torch.randperm(n, generator=gen)[: min(p, n)].sort().values
        x = torch.zeros(n, idx.numel(), dtype=DTYPE)
        x[idx, torch.arange(idx.numel())] = 1.0
        return x
    if kind == "rademacher":
        return (torch.randint(0, 2, (n, p), generator=gen).to(DTYPE) * 2.0) - 1.0
    if kind == "hadamard":
        size = 1
        while size < max(n, p):
            size *= 2
        h = torch.ones(1, 1, dtype=DTYPE)
        while h.shape[0] < size:
            h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
        return h[:n, :p].clone()
    raise ValueError(f"unknown probe kind {kind!r}")


def ls_gain(y_ref: torch.Tensor, y_now: torch.Tensor, dim: int | None = None) -> torch.Tensor:
    """Least-squares correction g minimising ||g * y_now - y_ref||^2 (per slice along ``dim``).

    dim=None -> scalar; dim=1 -> one gain per row of the (rows, P) arrays."""
    if dim is None:
        num, den = (y_ref * y_now).sum(), (y_now * y_now).sum()
    else:
        num, den = (y_ref * y_now).sum(dim), (y_now * y_now).sum(dim)
    return num / den.clamp(min=1e-300)


def apply_gains(
    w: torch.Tensor,
    alpha: float | torch.Tensor = 1.0,
    row: torch.Tensor | None = None,
    col: torch.Tensor | None = None,
) -> torch.Tensor:
    """diag(row) * alpha * W * diag(col): digital post/pre scaling of the analog operator."""
    out = w * alpha
    if row is not None:
        out = out * row[:, None]
    if col is not None:
        out = out * col[None, :]
    return out


class Bindable:
    """Mixin: truth providers used only to materialise W_eff (see module docstring)."""

    _w_now: Callable[[], torch.Tensor] | None = None
    _w0: Callable[[], torch.Tensor] | None = None

    def bind(self, w_now: Callable[[], torch.Tensor], w0: Callable[[], torch.Tensor] | None = None):
        self._w_now, self._w0 = w_now, w0
        return self

    def now(self) -> torch.Tensor:
        if self._w_now is None:
            raise RuntimeError(f"{getattr(self, 'name', type(self).__name__)}: call bind(w_now)")
        return self._w_now().to(DTYPE)

    def ref(self) -> torch.Tensor:
        if self._w0 is None:
            raise RuntimeError(f"{getattr(self, 'name', type(self).__name__)}: call bind(.., w0)")
        return self._w0().to(DTYPE)


def bind_all(methods, w_now, w0=None) -> None:
    for m in methods:
        if isinstance(m, Bindable):
            m.bind(w_now, w0)


def cost_since(reader: TileReaderLike, start: ReadCost) -> ReadCost:
    return reader.cost - start
