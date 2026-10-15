"""Phase H / H2 baselines: each must recover its own model class on a toy case."""

from __future__ import annotations

import numpy as np
import torch

from neurofault.spectral.encoding import encode
from neurofault.spectral.hardening import (
    HB3_GRID_S,
    RefColumnAffine,
    TimerLowRankAdapters,
    fit_channel_affine,
    fit_lowrank_adapter,
    hb1_calibrate,
    interleave_index,
    merge_rows,
    reference_weights,
    take_rows,
)
from neurofault.spectral.types import DTYPE, EncodingSpec, ReadCost, Tile, TileSpec, make_generator


def test_hb1_affine_recovers_channel_gain_bias():
    g = torch.Generator().manual_seed(0)
    y_ref = torch.randn(6, 500, generator=g, dtype=DTYPE)
    g_true, b_true = torch.rand(6, dtype=DTYPE) + 0.5, torch.randn(6, dtype=DTYPE)
    y_ana = (y_ref - b_true[:, None]) / g_true[:, None]  # y_ref = g_true * y_ana + b_true
    gh, bh = fit_channel_affine(y_ana, y_ref)
    assert torch.allclose(gh, g_true, atol=1e-9) and torch.allclose(bh, b_true, atol=1e-9)


def test_hb1_sequential_corrects_two_layer_chain():
    """Layer 2 sees the corrected layer 1; the chain output is restored to the digital one."""
    g = torch.Generator().manual_seed(1)
    w1, w2 = (torch.randn(8, 8, generator=g, dtype=DTYPE) for _ in range(2))
    x = torch.randn(8, 400, generator=g, dtype=DTYPE)
    s1, s2 = torch.rand(8, dtype=DTYPE) + 0.6, torch.rand(8, dtype=DTYPE) + 0.6  # per-channel drift
    a1, a2 = s1[:, None] * w1, s2[:, None] * w2

    def ref(layer):
        h1 = w1 @ x
        return h1 if layer == "l1" else w2 @ torch.relu(h1)

    def analog(layer, corr):
        h1 = a1 @ x
        if "l1" in corr:
            h1 = corr["l1"][0][:, None] * h1 + corr["l1"][1][:, None]
        return h1 if layer == "l1" else a2 @ torch.relu(h1)

    corr = hb1_calibrate(["l1", "l2"], analog, ref)
    assert torch.allclose(corr["l1"][0], 1.0 / s1, atol=1e-8)
    out = analog("l2", corr)
    out = corr["l2"][0][:, None] * out + corr["l2"][1][:, None]
    assert torch.allclose(out, ref("l2"), atol=1e-7)


class _FakeReader:
    """Noise-free reader over a time-varying (m, n) matrix, counting forward MVMs."""

    backward_supported = True

    def __init__(self, provider):
        self.provider, self.cost = provider, ReadCost()

    @property
    def shape(self):
        return tuple(self.provider().shape)

    def mvm(self, x):
        self.cost.mvm_fwd += x.shape[1]
        return self.provider() @ x


def test_hb2_recovers_smooth_row_gain_and_counts_cost():
    m, n, k = 64, 40, 4
    arr_idx, ref_idx, ref_pos = interleave_index(m, k)
    assert len(arr_idx) == m and len(set(arr_idx) | set(ref_idx)) == m + k
    g = torch.Generator().manual_seed(2)
    w = torch.randn(m, n, generator=g, dtype=DTYPE)
    w_ref = torch.randn(k, n, generator=g, dtype=DTYPE)
    aug = torch.zeros(m + k, n, dtype=DTYPE)
    aug[arr_idx], aug[ref_idx] = w, w_ref
    # drift: the needed correction gain is linear in the array-row coordinate (a smooth field the
    # reference lines can); drifted rows are scaled by 1/gain
    coord = torch.zeros(m + k, dtype=DTYPE)
    coord[arr_idx], coord[ref_idx] = (
        torch.arange(m, dtype=DTYPE),
        torch.as_tensor(ref_pos, dtype=DTYPE),
    )
    gain_true = 1.2 + 0.004 * coord
    field = 1.0 / gain_true
    state = {"scale": torch.ones(m + k, dtype=DTYPE)}
    reader = _FakeReader(lambda: state["scale"][:, None] * aug)
    hb2 = RefColumnAffine(ref_idx, arr_idx, ref_pos, p=12, seed=1)
    hb2.bind(lambda: field[arr_idx][:, None] * w, lambda: w)
    hb2.calibrate(reader)
    state["scale"] = field
    cor = hb2.update(reader, 1e5)
    assert cor.cost.mvm_fwd == 12  # P MVMs, reference lines ride along
    inner = slice(int(np.ceil(ref_pos[0])), int(np.floor(ref_pos[-1])) + 1)  # between outer lines
    assert torch.allclose(cor.W_eff[inner], w[inner], atol=1e-6)  # linear field, linear interp
    assert torch.allclose(cor.meta["offset"], torch.zeros(m, dtype=DTYPE), atol=1e-6)


def test_hb2_reference_weights_follow_tile_magnitudes_and_state_split_roundtrip():
    g = torch.Generator().manual_seed(3)
    w = torch.randn(32, 24, generator=g, dtype=DTYPE) * 0.1
    ref = reference_weights(w, 4, 24, make_generator(0, "r"))
    assert ref.shape == (4, 24) and ref.abs().max() <= w.abs().max() + 1e-12
    enc = EncodingSpec("diff_pair", g_min_frac=0.01)
    absmax = float(w.abs().max())
    spec = TileSpec(0, "x", 0, 0, 32, 24, 256)
    sa, sr = encode(Tile(spec, w), enc, absmax=absmax), encode(Tile(spec, ref), enc, absmax=absmax)
    arr_idx, ref_idx, _ = interleave_index(32, 4)
    aug = merge_rows(sa, sr, arr_idx, ref_idx)
    assert torch.allclose(take_rows(aug, arr_idx).weights(), w) and torch.allclose(
        take_rows(aug, ref_idx).weights(), ref
    )


def test_hb3_adapter_is_cov_weighted_optimum_and_timer_switches():
    g = torch.Generator().manual_seed(4)
    m, n, r = 20, 16, 4
    u = torch.randn(m, r, generator=g, dtype=DTYPE)
    d_true = u @ torch.randn(r, n, generator=g, dtype=DTYPE)  # rank-4 nominal drift error
    cov = torch.eye(n, dtype=DTYPE)
    b, a = fit_lowrank_adapter(d_true, cov, r)
    assert torch.allclose(b @ a, d_true, atol=1e-9)  # recovers its own model class exactly
    # higher-rank truth: error equals the Eckart-Young tail under the cov weighting
    d_full = torch.randn(m, n, generator=g, dtype=DTYPE)
    b, a = fit_lowrank_adapter(d_full, cov, r)
    tail = torch.linalg.svdvals(d_full)[r:].pow(2).sum()
    assert torch.isclose(((d_full - b @ a) ** 2).sum(), tail, rtol=1e-9)
    # timer: two grid levels with different D; update picks the nearest level and reads nothing
    grid = [1e3, 1e6]
    ds = [d_true * 0.1, d_true]
    hb3 = TimerLowRankAdapters(grid, rank=r).fit(ds, cov)
    w0 = torch.randn(m, n, generator=g, dtype=DTYPE)
    hb3.bind(lambda: w0 - d_true, lambda: w0)
    assert hb3.level(2e3) == 0 and hb3.level(3e5) == 1
    cor = hb3.update(_FakeReader(lambda: w0), 2e6)
    assert cor.cost.mvm_fwd == 0 and torch.allclose(cor.W_eff, w0, atol=1e-8)
    assert hb3.n_params() == 2 * r * (m + n) and len(HB3_GRID_S) == 11
    assert np.isclose(HB3_GRID_S[-2], 10**7.5)
