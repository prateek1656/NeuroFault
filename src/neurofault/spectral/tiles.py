"""Checkpoint loading, tiling and synthetic tiles.

Tiles are blocks of ``state_dict[name].flatten(1)`` - the same orientation the
feasibility probe and CrossSimAccessor.read() use - so a Tile's (r0, c0, m, n)
maps straight back onto the layer matrix when a corrected tile is written back.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

import torch

from neurofault.spectral.types import DTYPE, Tile, TileSpec, make_generator

REPO_ROOT = Path(__file__).resolve().parents[3]


def checkpoint_path(seed: int) -> Path:
    """Pre-deploy ResNet-9 checkpoint of a training seed (42, 101, 102 exist)."""
    return REPO_ROOT / "results" / "_live2" / f"resnet9_predeploy_seed{seed}" / "pre_deploy.pt"


def load_checkpoint(path_or_seed: str | Path | int) -> dict[str, torch.Tensor]:
    """{layer_name: flatten(1) float64 matrix} for every >=2-D weight.

    float64 because sigma_1/sigma_2 is ~1.02-1.11 on trained tiles and the
    spectral work downstream needs the precision (types.DTYPE)."""
    path = (
        checkpoint_path(int(path_or_seed)) if isinstance(path_or_seed, int) else Path(path_or_seed)
    )
    blob = torch.load(path, map_location="cpu", weights_only=False)
    sd = blob["model_state_dict"]
    return {k: v.detach().flatten(1).to(DTYPE) for k, v in sd.items() if v.ndim >= 2}


def extract_tiles(
    ckpt_seed: int,
    layers: Sequence[str] | None,
    tile_size: int,
    *,
    min_dim: int = 64,
    max_tiles_per_layer: int | None = None,
    tile_selection: Literal["first", "random", "all"] = "random",
    gen: torch.Generator | None = None,
) -> list[Tile]:
    """Cut layers into tile_size x tile_size blocks (raster order of origin).

    Edge blocks with either side < min_dim are skipped: a 10x512 fc strip has no
    meaningful spectral gap and is not a realistic physical tile either.
    'random' subsamples max_tiles_per_layer blocks with ``gen`` (default: a
    stream derived from ckpt_seed so the call is replayable); 'first' takes the
    first ones in raster order; 'all' ignores max_tiles_per_layer."""
    mats = load_checkpoint(ckpt_seed)
    names = list(mats) if layers is None else list(layers)
    if tile_selection == "random" and gen is None:
        gen = make_generator(ckpt_seed, "tiles")
    out: list[Tile] = []
    for name in names:
        W = mats[name]
        M, N = W.shape
        origins = [
            (r0, c0)
            for r0 in range(0, M, tile_size)
            for c0 in range(0, N, tile_size)
            if min(tile_size, M - r0) >= min_dim and min(tile_size, N - c0) >= min_dim
        ]
        if tile_selection != "all" and max_tiles_per_layer is not None:
            k = min(max_tiles_per_layer, len(origins))
            if tile_selection == "first":
                origins = origins[:k]
            else:
                idx = torch.randperm(len(origins), generator=gen)[:k].sort().values.tolist()
                origins = [origins[i] for i in idx]
        for r0, c0 in origins:
            m, n = min(tile_size, M - r0), min(tile_size, N - c0)
            spec = TileSpec(ckpt_seed, name, r0, c0, m, n, tile_size)
            out.append(Tile(spec, W[r0 : r0 + m, c0 : c0 + n].clone()))
    return out


def _orthonormal(rows: int, cols: int, gen: torch.Generator) -> torch.Tensor:
    """Haar-ish random orthonormal columns (QR with the sign ambiguity fixed)."""
    q, r = torch.linalg.qr(torch.randn(rows, cols, generator=gen, dtype=DTYPE))
    d = torch.diagonal(r).sign()
    d[d == 0] = 1.0
    return q * d


def synthetic_tile(
    m: int,
    n: int,
    *,
    spectrum: Literal["flat_gap1.05", "decay", "gaussian"],
    gen: torch.Generator,
    std: float = 0.05,
) -> Tile:
    """Checkpoint-free tile for CI. Entry std is normalised to ``std``.

    flat_gap1.05: sigma_1/sigma_2 = 1.05 exactly, remaining singular values
    spread linearly over [0.5, 1] x sigma_2 (the near-degenerate top mode of
    trained tiles); decay: sigma_i ~ 1/i; gaussian: iid N(0, std^2)."""
    r = min(m, n)
    if spectrum == "gaussian":
        w = std * torch.randn(m, n, generator=gen, dtype=DTYPE)
    else:
        if spectrum == "flat_gap1.05":
            s = torch.linspace(1.0, 0.5, max(r - 1, 1), dtype=DTYPE)  # s_2 .. s_r
            s = torch.cat([torch.tensor([1.05], dtype=DTYPE), s])[:r]
        elif spectrum == "decay":
            s = 1.0 / torch.arange(1, r + 1, dtype=DTYPE)
        else:
            raise ValueError(f"unknown spectrum {spectrum!r}")
        U = _orthonormal(m, r, gen)
        V = _orthonormal(n, r, gen)
        w = (U * s) @ V.T
        w = w * (std * math.sqrt(m * n) / torch.linalg.matrix_norm(w))
    return Tile(TileSpec(-1, f"synthetic_{spectrum}", 0, 0, m, n, max(m, n)), w)
