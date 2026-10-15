"""Phase H / H2 baselines (criteria/a_hardening.yaml): stronger calibration baselines against M4.

HB1  digital per-output-channel affine recalibration. A fixed set of N calibration images (augmented
     TRAIN images) is run through the deployed analog network layer by layer; for every analog layer
     the per-channel gain and bias are the least-squares fit of the analog output (sequentially, with
     the earlier layers already corrected) to the stored digital reference output of that layer.
     Runtime reads = N images x (per-image tile MVMs of the whole network); no probes.
HB2  per-column reference-column correction in the style of US 11663458: K reference output lines per
     tile (extra rows of W, programmed to known values drawn from the tile's own |w| distribution,
     interleaved with the array rows so they sit in the same die field) are read with P probes each
     recalibration and compared with their stored t0 reads; an affine gain/offset is fitted per
     reference line by least squares and the per-output gain is interpolated linearly between the
     reference lines. Runtime reads = P MVMs per tile (the reference lines ride along in every MVM);
     the price is K extra lines of area per tile.
HB3  VeRA+/LoRA-style open-loop adapters: offline-trained digital rank-r corrections B_g A_g for a grid
     of nominal drift times g, selected by a timer (nearest grid time in log(t + t0)); no on-chip
     sensing, 0 runtime reads. Training here is layer-wise reconstruction of the NOMINAL drift (D2) on
     training-set activations, in closed form (see fit_lowrank_adapter).

Pure functions / small classes that touch tiles only through TileReaderLike (HB2) or not at all (HB3).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence

import numpy as np
import torch

from neurofault.spectral.baselines.base import Bindable, make_probes
from neurofault.spectral.types import (
    DTYPE,
    T0_S,
    ConductanceState,
    Correction,
    ReadCost,
    TileReaderLike,
    make_generator,
)

# ----------------------------------------------------------------------------------------- HB1


def fit_channel_affine(
    y_ana: torch.Tensor, y_ref: torch.Tensor, eps: float = 1e-12
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-channel least squares y_ref ~ g * y_ana + b over samples. Inputs (C, S)."""
    y_ana, y_ref = y_ana.to(DTYPE), y_ref.to(DTYPE)
    ma, mr = y_ana.mean(1), y_ref.mean(1)
    da, dr = y_ana - ma[:, None], y_ref - mr[:, None]
    var = (da * da).mean(1)
    g = torch.where(var > eps, (da * dr).mean(1) / var.clamp(min=eps), torch.ones_like(var))
    return g, mr - g * ma


def hb1_calibrate(
    layers: Sequence[str],
    analog_out: Callable[[str, dict], torch.Tensor],
    ref_out: Callable[[str], torch.Tensor],
) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    """Sequential layer-wise affine recalibration.

    analog_out(layer, corr) -> (C, S) analog pre-bias outputs of `layer` with the corrections
    already fitted for earlier layers (corr: layer -> (g, b)) applied; ref_out(layer) -> (C, S)
    stored digital reference. Returns layer -> (g, b)."""
    corr: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    for layer in layers:
        corr[layer] = fit_channel_affine(analog_out(layer, corr), ref_out(layer))
    return corr


def hb1_mvm_equivalents(n_images: int, mvms_per_image: int) -> int:
    """HB1 cost: N images x tile-MVMs one image needs through the whole analog network."""
    return int(n_images) * int(mvms_per_image)


# ----------------------------------------------------------------------------------------- HB2


def interleave_index(m: int, k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Positions of k reference lines among m array rows in an (m+k)-row augmented tile.

    Returns (arr_idx (m,), ref_idx (k,), ref_pos (k,)) with ref_pos in ARRAY-row coordinates (a
    reference line sitting between array rows j-1 and j has position j - 0.5)."""
    total = m + k
    ref_idx = np.array([int((q + 0.5) * total / k) for q in range(k)])
    mask = np.ones(total, bool)
    mask[ref_idx] = False
    arr_idx = np.flatnonzero(mask)
    ref_pos = ref_idx - np.arange(k) - 0.5
    return arr_idx, ref_idx, ref_pos


def reference_weights(w_tile: torch.Tensor, k: int, n: int, gen: torch.Generator) -> torch.Tensor:
    """(k, n) known reference values: magnitudes follow the tile's own |w| distribution (evenly
    spaced quantiles, independently permuted per line) with random signs, so a reference line
    drifts like a typical array line (the drift exponent depends on the programmed conductance)."""
    mags = torch.sort(w_tile.abs().flatten()).values
    q = ((torch.arange(n, dtype=DTYPE) + 0.5) / n * (mags.numel() - 1)).long()
    base = mags[q]
    out = torch.empty(k, n, dtype=DTYPE)
    for i in range(k):
        perm = torch.randperm(n, generator=gen)
        sign = torch.randint(0, 2, (n,), generator=gen).to(DTYPE) * 2.0 - 1.0
        out[i] = base[perm] * sign
    return out


def take_rows(state: ConductanceState, idx: np.ndarray | torch.Tensor) -> ConductanceState:
    """Sub-state with the given rows (a view of the same die, same scale/encoding)."""
    ix = torch.as_tensor(idx, dtype=torch.long)
    return ConductanceState(
        g_pos=state.g_pos[ix].clone(),
        g_neg=None if state.g_neg is None else state.g_neg[ix].clone(),
        scale=state.scale,
        enc=state.enc,
        t=state.t,
        stuck=state.stuck[ix].clone(),
    )


def merge_rows(
    arr: ConductanceState, ref: ConductanceState, arr_idx: np.ndarray, ref_idx: np.ndarray
) -> ConductanceState:
    """Augmented (m+k)-row state: array rows at arr_idx, reference rows at ref_idx."""
    total = len(arr_idx) + len(ref_idx)
    ai, ri = torch.as_tensor(arr_idx), torch.as_tensor(ref_idx)
    n = arr.g_pos.shape[1]

    def cat(a: torch.Tensor, r: torch.Tensor, *extra: int) -> torch.Tensor:
        out = torch.zeros(total, n, *extra, dtype=a.dtype)
        out[ai], out[ri] = a, r
        return out

    return ConductanceState(
        g_pos=cat(arr.g_pos, ref.g_pos),
        g_neg=None if arr.g_neg is None else cat(arr.g_neg, ref.g_neg),
        scale=arr.scale,
        enc=arr.enc,
        t=arr.t,
        stuck=cat(arr.stuck, ref.stuck, 2),
    )


class RefColumnAffine(Bindable):
    """HB2. `reader` is a reader over the AUGMENTED tile (m+K rows, ref_idx are the reference lines).

    calibrate: P Rademacher probes, store the t0 reads of the reference lines (setup, not counted
    per update, like every other method's t0 reads). update: re-read the same probes (P MVMs, which
    read array and reference lines together), per reference line fit y0 ~ g y_now + o by least
    squares, and set the per-output gain by piecewise-linear interpolation of the line gains over
    the array-row coordinate (constant extrapolation beyond the outer lines). W_eff = diag(g) W(t).
    The fitted offsets are returned in meta; the experiment uses the shared mean-bias read for the
    output offset so that every method gets the same bias treatment."""

    def __init__(
        self,
        ref_idx: np.ndarray,
        arr_idx: np.ndarray,
        ref_pos: np.ndarray,
        p: int = 15,
        seed: int = 0,
    ):
        self.ref_idx, self.arr_idx, self.ref_pos, self.p = ref_idx, arr_idx, ref_pos, p
        self.gen = make_generator(seed, "hb2_probes")
        self.name = f"HB2_refcols{len(ref_idx)}_p{p}"
        self.x: torch.Tensor | None = None
        self.y0: torch.Tensor | None = None

    def calibrate(self, reader: TileReaderLike) -> ReadCost:
        start = reader.cost.copy()
        self.x = make_probes("rademacher", reader.shape[1], self.p, self.gen)
        self.y0 = reader.mvm(self.x)[self.ref_idx]
        return reader.cost - start

    def update(self, reader: TileReaderLike, t: float) -> Correction:
        assert self.x is not None, "calibrate() first"
        start = reader.cost.copy()
        yt = reader.mvm(self.x)[self.ref_idx]
        gk, ok = fit_channel_affine(yt, self.y0)
        rows = np.arange(len(self.arr_idx))
        g = torch.as_tensor(np.interp(rows, self.ref_pos, gk.numpy()), dtype=DTYPE)
        o = torch.as_tensor(np.interp(rows, self.ref_pos, ok.numpy()), dtype=DTYPE)
        w = self.now() * g[:, None]
        return Correction(w, reader.cost - start, {"row_gain": g, "offset": o, "ref_gain": gk})


# ----------------------------------------------------------------------------------------- HB3


def fit_lowrank_adapter(
    d_bar: torch.Tensor, cov: torch.Tensor, rank: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rank-r adapter (B (m, r), A (r, n)) minimising E_x ||(D - B A) x||^2 with E[x x^T] = cov.

    This is the exact minimiser of the layer-wise reconstruction loss a LoRA branch is trained on
    (the expected drift error D = W0 - E[W(t)] against the training-set input statistics):
    with U S V^T = svd(D cov^(1/2)), the optimum is B A = U_r S_r V_r^T cov^(-1/2) = U_r U_r^T D, so no
    inverse square root is needed. B = U_r, A = U_r^T D."""
    ev, vec = torch.linalg.eigh((cov + cov.T) / 2)
    c_half = (vec * ev.clamp_min(0).sqrt()) @ vec.T
    u = torch.linalg.svd(d_bar @ c_half, full_matrices=False)[0][:, :rank]
    return u, u.T @ d_bar


class TimerLowRankAdapters(Bindable):
    """HB3: per-grid-time rank-r adapters, chosen by elapsed time only. 0 runtime reads."""

    def __init__(self, grid_s: Sequence[float], rank: int = 4, t0: float = T0_S):
        self.grid = [float(g) for g in grid_s]
        self.rank, self.t0 = rank, t0
        self.name = f"HB3_timer_lora_r{rank}_g{len(self.grid)}"
        self.adapters: list[tuple[torch.Tensor, torch.Tensor]] = []
        self.alphas: list[float] = []

    def fit(
        self,
        d_bar_by_level: Sequence[torch.Tensor],
        cov: torch.Tensor,
        alphas: Sequence[float] | None = None,
    ) -> TimerLowRankAdapters:
        """d_bar_by_level[g] = residual E[W0 - alpha_g W_nom(t_g)] after the nominal scalar gain
        alpha_g (a purely additive rank-r branch cannot express a multiplicative decay of a
        full-rank matrix, so each level also carries the open-loop timer gain, as in B6/GDC-with-LoRA
        deployments; alphas=None means 1)."""
        self.alphas = [1.0] * len(d_bar_by_level) if alphas is None else [float(a) for a in alphas]
        self.adapters = [fit_lowrank_adapter(d, cov, self.rank) for d in d_bar_by_level]
        return self

    def level(self, t: float) -> int:
        lt = math.log(t + self.t0)
        return min(range(len(self.grid)), key=lambda i: abs(math.log(self.grid[i] + self.t0) - lt))

    def n_params(self) -> int:
        return sum(b.numel() + a.numel() for b, a in self.adapters)

    def calibrate(self, reader: TileReaderLike) -> ReadCost:
        return ReadCost()

    def update(self, reader: TileReaderLike, t: float) -> Correction:
        lv = self.level(t)
        b, a = self.adapters[lv]
        w = self.alphas[lv] * self.now() + b @ a
        return Correction(
            w, ReadCost(), {"level": lv, "grid_t": self.grid[lv], "alpha": self.alphas[lv]}
        )


# grid used by the experiment: two levels per decade from 1e3 s to 1e8 s
HB3_GRID_S = tuple(float(v) for e in range(3, 8) for v in (10.0**e, 10.0 ** (e + 0.5))) + (1e8,)
