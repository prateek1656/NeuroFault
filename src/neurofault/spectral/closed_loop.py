"""Closed loop X.1/X.2: estimate the stuck-cell map of a deployed CrossSim system from counted,
noisy probe reads, so the mitigation controller (remap + bursts) acts on an ESTIMATE, not on the
injector's truth (the deferred closed-loop step).

The estimator sees each layer only through per-tile ``TileReader``s: noisy forward MVMs of the
layer's CURRENT conductance matrix (``handle.accessor.read()`` stays inside the reader shim and
is never handed to a localizer). The reference is what a real controller knows: its digital
weights plus the deviations it already believes for cells it has flagged but not yet repaired.

Locators (per 256x256 tile, partial edge tiles kept):
- ``readout``: A1 full readout (n unit-vector reads), fixed-false-alarm threshold.
- ``adaptive`` (ours): A5 forward compressed sensing (P dense Rademacher probes, joint OMP);
  if the OMP hits its support cap the tile is too dense for sparse recovery and falls back to
  A1 on that tile (B E1c: probing beats full readout only below ~0.3% density).
``shadow`` runs the other locator on the same state for cost/accuracy logging only.

Noise: white per-cell read noise sigma_frac * max|w| of the tile (H8 regime), sigma passed in
(calibration read cost is charged separately as 2*n_calib reads per tile when ``charge_calib``).
"""

from __future__ import annotations

import zlib
from dataclasses import dataclass, field

import torch

from neurofault.spectral.localize.common import z_two_sided
from neurofault.spectral.localize.cs import NonAdaptiveCSLocalizer
from neurofault.spectral.localize.full_readout import FullReadoutLocalizer
from neurofault.spectral.readout import TileReader
from neurofault.spectral.types import DTYPE, ReadNoise, make_generator


class _MatrixState:
    """Minimal ConductanceState stand-in for TileReader: a fixed weight-space block."""

    def __init__(self, w: torch.Tensor) -> None:
        self._w = w
        self.t = 0.0

    @property
    def shape(self) -> tuple[int, int]:
        return tuple(self._w.shape)

    def weights(self) -> torch.Tensor:
        return self._w


class _CS(NonAdaptiveCSLocalizer):
    """A5 that also keeps the last OMP support and its (whitened) coefficients."""

    def localize(self, reader, t: float = 0.0):
        self.support: dict[tuple[int, int], float] = {}
        self.n_selected = 0
        return super().localize(reader, t)

    def _refit(self, ri, cj, Kf, Kb, Zf0, Xc, Zb0, Yr, zf, zb):
        e = super()._refit(ri, cj, Kf, Kb, Zf0, Xc, Zb0, Yr, zf, zb)
        self.n_selected = len(ri)
        self.support = {(i, j): float(v) for i, j, v in zip(ri, cj, e.tolist(), strict=True)}
        return e


def _readout(reader: TileReader, w_ref: torch.Tensor, sigma: float, fa: float):
    """A1 with the signed residual kept: (cells (K,2), signed deviation (K,), reads)."""
    loc = FullReadoutLocalizer(w_ref, sigma=sigma, fa=fa)
    before = reader.cost.copy()
    E = loc.res_fwd(reader, torch.eye(w_ref.shape[1], dtype=DTYPE))
    mask = E.abs() > z_two_sided(E.numel(), fa) * sigma
    cells = mask.nonzero()
    cost = reader.cost - before
    return cells, E[mask], cost.mvm_fwd + cost.mvm_bwd


def _cs(reader, w_ref, sigma, fa, p, cap, seed):
    """A5 forward: (cells, signed deviation, reads, cap_hit)."""
    loc = _CS(w_ref, sigma=sigma, fa=fa, p_fwd=p, p_bwd=0, weight=None, max_nnz=cap, seed=seed)
    est, cost = loc.localize(reader)
    cells = est.cells
    dev = torch.tensor(
        [loc.support.get((int(r), int(c)), 0.0) * sigma for r, c in cells.tolist()], dtype=DTYPE
    )
    return cells, dev, cost.mvm_fwd + cost.mvm_bwd, loc.n_selected >= cap


@dataclass
class LayerEstimate:
    mask: torch.Tensor  # (rows, cols) bool, newly believed this call
    deviation: torch.Tensor  # (rows, cols) signed estimate, valid where mask
    reads: int = 0
    tiles: int = 0
    fallback_tiles: int = 0
    shadow: dict = field(default_factory=dict)


class ProbeEstimator:
    """Callable used by NeuromorphicSystem.localize_now(): (layer name, sealed matrix provider,
    digital reference, call index) -> LayerEstimate."""

    def __init__(
        self,
        kind: str = "adaptive",
        sigma_frac: float = 0.02,
        tile: int = 256,
        p_cs: int = 24,
        cap: int = 64,
        fa: float = 0.1,
        seed: int = 0,
        shadow: bool = True,
        charge_calib: int = 4,
    ) -> None:
        if kind not in ("adaptive", "readout"):
            raise ValueError(f"unknown locator kind {kind!r}")
        self.kind, self.sigma_frac, self.tile = kind, sigma_frac, tile
        self.p_cs, self.cap, self.fa, self.seed = p_cs, cap, fa, seed
        self.shadow, self.charge_calib = shadow, charge_calib

    def describe(self) -> dict:
        return {k: getattr(self, k) for k in ("kind", "sigma_frac", "tile", "p_cs", "cap", "fa",
                                              "seed", "shadow", "charge_calib")}  # fmt: skip

    def _run(self, kind, w_now, w_ref, sigma, stream):
        reader = TileReader(lambda: _MatrixState(w_now), ReadNoise(mode="per_cell",
                            sigma_cell_frac=self.sigma_frac), make_generator(self.seed, stream),
                            backward_supported=False)  # fmt: skip
        if kind == "readout":
            cells, dev, reads = _readout(reader, w_ref, sigma, self.fa)
            return cells, dev, reads + 2 * self.charge_calib, False
        cells, dev, reads, hit = _cs(reader, w_ref, sigma, self.fa, self.p_cs, self.cap,
                                     self.seed * 1_000_003 + zlib.crc32(stream.encode()))  # fmt: skip
        reads += 2 * self.charge_calib
        if hit:  # too dense for sparse recovery: fall back to full readout on this tile
            c2, d2, r2, _ = self._run("readout", w_now, w_ref, sigma, stream + ":fb")
            return c2, d2, reads + r2, True
        return cells, dev, reads, False

    @torch.no_grad()
    def __call__(self, name: str, read_matrix, w_ref: torch.Tensor, call: int) -> LayerEstimate:
        w_now_full = read_matrix().detach().to("cpu", DTYPE)
        ref_full = w_ref.detach().to("cpu", DTYPE)
        R, C = ref_full.shape
        mask = torch.zeros(R, C, dtype=torch.bool)
        dev = torch.zeros(R, C, dtype=DTYPE)
        out = LayerEstimate(mask, dev)
        sh = {"reads": 0, "cells": []}
        other = "readout" if self.kind == "adaptive" else "adaptive"
        for r0 in range(0, R, self.tile):
            for c0 in range(0, C, self.tile):
                w_now = w_now_full[r0 : r0 + self.tile, c0 : c0 + self.tile]
                ref = ref_full[r0 : r0 + self.tile, c0 : c0 + self.tile]
                sigma = self.sigma_frac * float(w_now.abs().max()) or 1e-12
                stream = f"xloop:{name}:{call}:{r0}:{c0}"
                cells, d, reads, fb = self._run(self.kind, w_now, ref, sigma, stream)
                if cells.numel():
                    mask[cells[:, 0] + r0, cells[:, 1] + c0] = True
                    dev[cells[:, 0] + r0, cells[:, 1] + c0] = d
                out.reads += reads
                out.tiles += 1
                out.fallback_tiles += int(fb)
                if self.shadow:
                    c2, _d2, r2, _ = self._run(other, w_now, ref, sigma, stream + ":shadow")
                    sh["reads"] += r2
                    if c2.numel():
                        sh["cells"].append(c2 + torch.tensor([r0, c0]))
        if self.shadow:
            smask = torch.zeros(R, C, dtype=torch.bool)
            for c2 in sh["cells"]:
                smask[c2[:, 0], c2[:, 1]] = True
            out.shadow = {"kind": other, "reads": sh["reads"], "mask": smask}
        return out
