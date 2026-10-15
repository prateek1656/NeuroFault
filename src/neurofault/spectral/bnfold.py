"""Helpers for the BN-folded field-fit follow-up (criteria/a_bnfold.yaml). Pure functions.

Tile enumeration from a matrix dict (so a CIFAR-100 checkpoint tiles the same way as the
CIFAR-10 one), and the pooled paired cluster-bootstrap gain CI the registered criteria need.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd
import torch

from neurofault.spectral.exp2 import cluster_bootstrap_ci

SEED_OFFSET = 1000  # all random streams are shifted by this vs Exp 2 (a_bnfold.yaml note)


def tile_origins(
    mats: Mapping[str, torch.Tensor], layers: Sequence[str], tile_size: int, min_dim: int = 16
) -> list[tuple[str, int, int, int, int]]:
    """(layer, r0, c0, m, n) for every tile_size block of each layer matrix in raster order;
    edge blocks with a side < min_dim are skipped (same rule as tiles.extract_tiles 'all')."""
    out = []
    for name in layers:
        big_m, big_n = mats[name].shape
        for r0 in range(0, big_m, tile_size):
            for c0 in range(0, big_n, tile_size):
                m, n = min(tile_size, big_m - r0), min(tile_size, big_n - c0)
                if min(m, n) >= min_dim:
                    out.append((name, r0, c0, m, n))
    return out


def add_gain_ci(
    df: pd.DataFrame, n_boot: int = 4000, seed: int = 0, alpha: float = 0.05
) -> pd.DataFrame:
    """Add gain_ci_lower_pt: per (dataset, t_s), the lower bound of the (1-alpha) cluster
    bootstrap CI of gain_m4_minus_b8_pt pooled over scenarios, resampling ckpt seeds as
    clusters. The same value is repeated on every row of the group."""
    df = df.copy()
    df["gain_ci_lower_pt"] = np.nan
    for idx in df.groupby(["dataset", "t_s"]).groups.values():
        g = df.loc[idx]
        _, lo, _ = cluster_bootstrap_ci(
            g.gain_m4_minus_b8_pt.to_numpy(), g.ckpt_seed.to_numpy(), n_boot, alpha, seed
        )
        df.loc[idx, "gain_ci_lower_pt"] = lo
    return df
