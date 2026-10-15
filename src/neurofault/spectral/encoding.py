"""Weights <-> conductances.

Conventions follow AIHWKit's SinglePairConductanceConverter: the tile's abs-max
weight maps to g_max, a positive weight sits on G+ and a negative one on G-, the
idle device rests at g_min. Because ConductanceState.weights() divides by the
same scale, encode -> weights() is an exact round trip (to float rounding).
"""

from __future__ import annotations

from neurofault.spectral.types import DTYPE, ConductanceState, EncodingSpec, Tile


def encode(tile: Tile, enc: EncodingSpec, *, absmax: float | None = None) -> ConductanceState:
    """Encode a tile's signed weights as a physical ConductanceState (t = 0).

    The scale is set so ``absmax`` fills the usable conductance range:
    tile_absmax uses max|tile.w|; layer_absmax requires the caller to pass the
    layer's abs-max via ``absmax`` (it cannot be known from one tile). An explicit
    ``absmax`` is also accepted under tile_absmax and wins."""
    w = tile.w.to(DTYPE)
    if absmax is None:
        if enc.scale_policy == "layer_absmax":
            raise ValueError("layer_absmax needs an explicit absmax= (the layer's max |w|)")
        absmax = float(w.abs().max())
    if absmax <= 0.0:
        absmax = 1.0  # all-zero tile: any positive scale decodes back to zeros
    g_min, g_max = enc.g_min_uS, enc.g_max_uS
    if enc.kind == "diff_pair":
        scale = (g_max - g_min) / absmax
        g_pos = g_min + w.clamp(min=0.0) * scale
        g_neg = g_min + (-w).clamp(min=0.0) * scale
        return ConductanceState(g_pos, g_neg, scale, enc)
    if enc.kind == "offset_refcol":
        scale = 0.5 * (g_max - g_min) / absmax
        g_ref = 0.5 * (g_min + g_max)
        return ConductanceState(g_ref + w * scale, None, scale, enc)
    if enc.kind == "signed_ideal":
        scale = g_max / absmax
        return ConductanceState(w * scale, None, scale, enc)
    raise ValueError(f"unknown encoding {enc.kind!r}")


__all__ = ["encode"]
