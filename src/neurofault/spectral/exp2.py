"""Helpers for the Direction-A Exp 2 comparison against AdaBS (S2).

Pure functions, no I/O: BN folding (the BN-free workload W2), calibration-cost accounting in
MVM-equivalents, cluster-paired bootstrap, Pareto front. The experiment script owns the rest.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence

import numpy as np
import torch


def fold_conv_bn(
    w: torch.Tensor,
    gamma: torch.Tensor,
    beta: torch.Tensor,
    mean: torch.Tensor,
    var: torch.Tensor,
    eps: float = 1e-5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fold an inference-mode BatchNorm into the preceding bias-free conv (out, ...).

    BN(conv(x)) = conv(x; w') + c with w' = w * s, c = beta - s * mean, s = gamma/sqrt(var+eps).
    The row scale s is part of the analog weights after folding (that is the point: deployed
    BN-free hardware stores w'), c stays a digital bias."""
    s = gamma / torch.sqrt(var + eps)
    return w * s.reshape(-1, *([1] * (w.ndim - 1))), beta - s * mean


def adabs_images_per_tile(positions: Sequence[int], tiles: Sequence[int]) -> int:
    """MVMs one calibration image costs: sum over layers of (tiles in layer x patch positions)."""
    if len(positions) != len(tiles):
        raise ValueError("positions and tiles must align")
    return int(sum(p * t for p, t in zip(positions, tiles, strict=True)))


def adabs_mvm_equivalents(n_images: int, mvms_per_image: int) -> int:
    """AdaBS cost in MVM-equivalents = images x per-image tile-MVMs ('Matched budget')."""
    return int(n_images) * int(mvms_per_image)


def cluster_bootstrap_ci(
    values: np.ndarray,
    clusters: np.ndarray,
    n_boot: int = 4000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Mean and (1-alpha) percentile CI of paired differences, resampling whole clusters
    (checkpoint seeds) with replacement and then rows inside each drawn cluster.

    With few clusters (3 seeds) the CI is wide by design; that is the honest answer."""
    values = np.asarray(values, float)
    clusters = np.asarray(clusters)
    ids = np.unique(clusters)
    groups = [values[clusters == c] for c in ids]
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.integers(0, len(ids), len(ids))
        parts = [groups[i][rng.integers(0, len(groups[i]), len(groups[i]))] for i in pick]
        means[b] = np.concatenate(parts).mean()
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return float(values.mean()), float(lo), float(hi)


def pareto_front(points: Iterable[tuple[float, float]]) -> list[int]:
    """Indices of non-dominated points (lower cost, higher accuracy), cost ascending."""
    pts = list(points)
    order = sorted(range(len(pts)), key=lambda i: (pts[i][0], -pts[i][1]))
    front, best = [], -math.inf
    for i in order:
        if pts[i][1] > best:
            front.append(i)
            best = pts[i][1]
    return front


def split_batches(n: int, max_bs: int = 260) -> list[int]:
    """Equal batch sizes covering n images with batches <= max_bs (AdaBS cumulative mean of
    per-batch statistics weights batches equally, so equal sizes avoid a biased average)."""
    nb = math.ceil(n / max_bs)
    base = n // nb
    sizes = [base + (1 if i < n - base * nb else 0) for i in range(nb)]
    assert sum(sizes) == n
    return sizes
