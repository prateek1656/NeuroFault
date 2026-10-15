"""Tiles, encoding and noise (item 1)."""

from __future__ import annotations

import pytest
import torch

from neurofault.spectral.encoding import encode
from neurofault.spectral.noise import PCM_PROG_COEFF, one_over_f_sigma, program, read_noise_cell
from neurofault.spectral.tiles import (
    checkpoint_path,
    extract_tiles,
    load_checkpoint,
    synthetic_tile,
)
from neurofault.spectral.types import DTYPE, EncodingSpec, ProgNoise, ReadNoise, make_generator


def _tile(m=48, n=40, seed=0, spectrum="gaussian"):
    return synthetic_tile(m, n, spectrum=spectrum, gen=make_generator(seed, "tiles"))


@pytest.mark.parametrize("kind", ["diff_pair", "offset_refcol", "signed_ideal"])
@pytest.mark.parametrize("gmin", [0.0, 0.01, 0.05])
def test_round_trip(kind, gmin):
    t = _tile()
    st = encode(t, EncodingSpec(kind=kind, g_min_frac=gmin))
    assert (st.weights() - t.w).abs().max() < 1e-12
    if kind != "signed_ideal":
        lo, hi = st.enc.g_min_uS, st.enc.g_max_uS
        for g in (st.g_pos, st.g_neg):
            if g is not None:
                assert g.min() >= lo - 1e-12 and g.max() <= hi + 1e-12


def test_diff_pair_layout_and_scale():
    t = _tile()
    enc = EncodingSpec(g_min_frac=0.05)
    st = encode(t, enc)
    pos = t.w > 0
    assert torch.allclose(st.g_neg[pos], torch.full_like(st.g_neg[pos], enc.g_min_uS))
    assert torch.allclose(st.g_pos[~pos], torch.full_like(st.g_pos[~pos], enc.g_min_uS))
    assert (
        abs(float(st.g_pos.max()) - enc.g_max_uS) < 1e-9
        or abs(float(st.g_neg.max()) - enc.g_max_uS) < 1e-9
    )


def test_layer_absmax():
    t = _tile()
    enc = EncodingSpec(scale_policy="layer_absmax")
    with pytest.raises(ValueError):
        encode(t, enc)
    big = float(t.w.abs().max()) * 4
    st = encode(t, enc, absmax=big)
    assert float(st.g_pos.max()) < enc.g_max_uS * 0.5
    assert (st.weights() - t.w).abs().max() < 1e-12


def test_zero_tile():
    t = _tile()
    t.w.zero_()
    assert encode(t, EncodingSpec()).weights().abs().max() == 0


def test_synthetic_gap_and_shape():
    t = synthetic_tile(128, 96, spectrum="flat_gap1.05", gen=make_generator(1, "tiles"))
    assert t.w.dtype == DTYPE and t.w.shape == (128, 96) and t.spec.ckpt_seed == -1
    s = torch.linalg.svdvals(t.w)
    assert abs(float(s[0] / s[1]) - 1.05) < 1e-9
    d = synthetic_tile(64, 64, spectrum="decay", gen=make_generator(1, "tiles"))
    sd = torch.linalg.svdvals(d.w)
    assert abs(float(sd[0] / sd[1]) - 2.0) < 1e-9
    g = synthetic_tile(200, 200, spectrum="gaussian", gen=make_generator(1, "tiles"))
    assert abs(float(g.w.std()) - 0.05) < 0.003


def test_determinism():
    a = _tile(seed=3, spectrum="flat_gap1.05")
    b = _tile(seed=3, spectrum="flat_gap1.05")
    assert torch.equal(a.w, b.w)
    st = encode(a, EncodingSpec())
    p1 = program(st, ProgNoise("aihwkit_pcm_poly"), make_generator(5, "prog"))
    p2 = program(st, ProgNoise("aihwkit_pcm_poly"), make_generator(5, "prog"))
    p3 = program(st, ProgNoise("aihwkit_pcm_poly"), make_generator(6, "prog"))
    assert torch.equal(p1.g_pos, p2.g_pos) and not torch.equal(p1.g_pos, p3.g_pos)


@pytest.mark.parametrize("kind", ["gauss_rel", "aihwkit_pcm_poly"])
def test_program_leaves_stuck_untouched_and_input_unmodified(kind):
    st = encode(_tile(), EncodingSpec(g_min_frac=0.01))
    st.stuck[::3, ::2, 0] = True
    st.stuck[1::4, ::5, 1] = True
    before_pos, before_neg = st.g_pos.clone(), st.g_neg.clone()
    out = program(st, ProgNoise(kind, rel_sigma=0.05), make_generator(0, "prog"))
    assert torch.equal(st.g_pos, before_pos) and torch.equal(st.g_neg, before_neg)
    assert torch.equal(out.g_pos[st.stuck[..., 0]], before_pos[st.stuck[..., 0]])
    assert torch.equal(out.g_neg[st.stuck[..., 1]], before_neg[st.stuck[..., 1]])
    free = ~st.stuck[..., 0]
    assert (out.g_pos[free] != before_pos[free]).float().mean() > 0.9
    assert out.g_pos.min() >= 0 and out.g_neg.min() >= 0


def test_program_none_is_identity():
    st = encode(_tile(), EncodingSpec())
    out = program(st, ProgNoise("none"), make_generator(0, "prog"))
    assert torch.equal(out.g_pos, st.g_pos) and out.g_pos is not st.g_pos


def test_gauss_rel_std():
    st = encode(_tile(300, 300), EncodingSpec())
    st.g_pos.fill_(12.5)  # away from the clamp
    out = program(st, ProgNoise("gauss_rel", rel_sigma=0.03), make_generator(0, "prog"))
    assert abs(float((out.g_pos - 12.5).std()) - 0.03 * 25.0) < 0.02


@pytest.mark.parametrize("gmax", [25.0, 50.0])
def test_pcm_poly_std(gmax):
    """sigma(g) = (c0 + c1 x + c2 x^2) * g_max/25, x = g/g_max (AIHWKit pcm.py)."""
    st = encode(_tile(400, 400), EncodingSpec(g_max_uS=gmax))
    x = 0.4
    st.g_pos.fill_(x * gmax)
    c0, c1, c2 = PCM_PROG_COEFF
    want = (c0 + c1 * x + c2 * x * x) * gmax / 25.0
    out = program(st, ProgNoise("aihwkit_pcm_poly"), make_generator(0, "prog"))
    assert abs(float((out.g_pos - x * gmax).std()) / want - 1) < 0.02


def test_one_over_f_formula():
    g = torch.tensor([0.25, 2.5, 25.0, 1e-6], dtype=DTYPE)
    s = one_over_f_sigma(g, 25.0, t=100.0)
    import math

    scale = math.sqrt(math.log((120.0 + 250e-9) / 500e-9))
    q = torch.tensor([0.0088 / 0.01**0.65, 0.0088 / 0.1**0.65, 0.0088, 0.2], dtype=DTYPE)
    assert torch.allclose(s, q.clamp(max=0.2) * scale)


def test_read_noise_cell_std_and_identity():
    st = encode(_tile(300, 300), EncodingSpec())
    w = st.weights()
    same = read_noise_cell(st, ReadNoise(mode="output_referred"), 0.0, make_generator(0, "r"))
    assert torch.equal(same, w)
    nz = read_noise_cell(st, ReadNoise(sigma_cell_frac=0.02), 0.0, make_generator(0, "r"))
    want = 0.02 * float(w.abs().max())
    assert abs(float((nz - w).std()) / want - 1) < 0.02
    f = read_noise_cell(
        st, ReadNoise(mode="output_referred", one_over_f=True), 1e4, make_generator(0, "r")
    )
    assert (f - w).abs().max() > 0


# --------------------------------------------------------------------- checkpoints


def _ckpt_or_skip(seed=42):
    if not checkpoint_path(seed).exists():
        pytest.skip("checkpoint not available")


def test_load_checkpoint_and_extract_tiles():
    _ckpt_or_skip()
    mats = load_checkpoint(42)
    assert "res1.conv_a.weight" in mats and mats["res1.conv_a.weight"].shape == (128, 1152)
    assert all(m.dtype == DTYPE and m.ndim == 2 for m in mats.values())
    tiles = extract_tiles(42, ["res1.conv_a.weight", "fc.weight"], 128, tile_selection="all")
    # fc is 10x512: every block < min_dim rows -> skipped; res1.conv_a: 1 x 9 blocks of 128x128
    assert len(tiles) == 9 and all(t.spec.layer == "res1.conv_a.weight" for t in tiles)
    t = tiles[3]
    s = t.spec
    assert torch.equal(t.w, mats[s.layer][s.r0 : s.r0 + s.m, s.c0 : s.c0 + s.n])
    # edge tile smaller than min_dim skipped: 1152 = 4*256 + 128 -> last col block is 128 wide
    e = extract_tiles(42, ["res1.conv_a.weight"], 256, min_dim=200, tile_selection="all")
    assert all(x.spec.n >= 200 and x.spec.m >= 200 for x in e)


def test_extract_tiles_selection_is_replayable():
    _ckpt_or_skip()
    L = ["layer3.0.0.weight"]
    a = extract_tiles(42, L, 128, max_tiles_per_layer=4, gen=make_generator(0, "tiles"))
    b = extract_tiles(42, L, 128, max_tiles_per_layer=4, gen=make_generator(0, "tiles"))
    first = extract_tiles(42, L, 128, max_tiles_per_layer=4, tile_selection="first")
    assert [x.spec for x in a] == [x.spec for x in b] and len(a) == 4
    assert [(x.spec.r0, x.spec.c0) for x in first] == [(0, 0), (0, 128), (0, 256), (0, 384)]
