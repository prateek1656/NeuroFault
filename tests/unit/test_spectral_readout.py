"""TileReader: cost counting, noise statistics, flags (item 1)."""

from __future__ import annotations

import pytest
import torch

from neurofault.spectral.encoding import encode
from neurofault.spectral.readout import TileReader, quantize
from neurofault.spectral.tiles import synthetic_tile
from neurofault.spectral.types import (
    DTYPE,
    EncodingSpec,
    ReadCost,
    ReadNoise,
    TileReaderLike,
    make_generator,
)

QUIET = ReadNoise(mode="per_cell", sigma_cell_frac=0.0, adc_bits=None, dac_bits=None)


def _state(m=40, n=30):
    t = synthetic_tile(m, n, spectrum="flat_gap1.05", gen=make_generator(0, "tiles"))
    return t, encode(t, EncodingSpec())


def _reader(noise=QUIET, seed=0, **kw):
    t, st = _state()
    return t, TileReader(lambda: st, noise, make_generator(seed, "read"), **kw)


def test_protocol_and_shape():
    _, r = _reader()
    assert isinstance(r, TileReaderLike) and r.shape == (40, 30)


def test_noiseless_mvm_matches_matrix():
    t, r = _reader()
    x = torch.randn(30, 5, dtype=DTYPE)
    y = torch.randn(40, 3, dtype=DTYPE)
    assert (r.mvm(x) - t.w @ x).abs().max() < 1e-12
    assert (r.mvm_t(y) - t.w.T @ y).abs().max() < 1e-12
    assert (r.full_readout() - t.w).abs().max() < 1e-12


def test_cost_counter_exact():
    _, r = _reader(noise=ReadNoise(adc_bits=8, dac_bits=8))
    r.mvm(torch.randn(30, 7, dtype=DTYPE))
    assert r.cost == ReadCost(mvm_fwd=7, adc=40 * 7, dac=30 * 7)
    r.mvm_t(torch.randn(40, 2, dtype=DTYPE))
    assert r.cost == ReadCost(mvm_fwd=7, mvm_bwd=2, adc=40 * 7 + 30 * 2, dac=30 * 7 + 40 * 2)
    r.full_readout()
    assert r.cost.cell_reads == 40 * 30
    assert r.cost.writes == 0 and r.cost.images == 0


def test_external_cost_object_accumulates():
    shared = ReadCost()
    _, r = _reader(cost=shared)
    r.mvm(torch.ones(30, 1, dtype=DTYPE))
    assert shared.mvm_fwd == 1 and r.cost is shared


def test_backward_flag():
    _, r = _reader(backward_supported=False)
    assert r.backward_supported is False
    with pytest.raises(RuntimeError):
        r.mvm_t(torch.ones(40, 1, dtype=DTYPE))
    assert r.cost.mvm_bwd == 0  # nothing counted for a refused call


def test_determinism_and_fresh_noise():
    nz = ReadNoise(sigma_cell_frac=0.02, adc_bits=None, dac_bits=None)
    x = torch.randn(30, 4, dtype=DTYPE)
    _, a = _reader(nz, seed=1)
    _, b = _reader(nz, seed=1)
    _, c = _reader(nz, seed=2)
    ya1, ya2 = a.mvm(x), a.mvm(x)
    assert torch.equal(ya1, b.mvm(x)) and not torch.equal(ya1, c.mvm(x))
    assert not torch.equal(ya1, ya2)  # fresh each read


def test_per_cell_noise_std():
    """Output noise std = sigma_cell * max|w| * ||x|| per output (white per-cell model)."""
    t, r = _reader(ReadNoise(sigma_cell_frac=0.02, adc_bits=None, dac_bits=None))
    x = torch.randn(30, 1, dtype=DTYPE).repeat(1, 4000)
    err = r.mvm(x) - (t.w @ x)
    want = 0.02 * float(t.w.abs().max()) * float(x[:, 0].norm())
    assert abs(float(err.std()) / want - 1) < 0.03
    # backward uses ||y||
    y = torch.randn(40, 1, dtype=DTYPE).repeat(1, 4000)
    errt = r.mvm_t(y) - t.w.T @ y
    wantt = 0.02 * float(t.w.abs().max()) * float(y[:, 0].norm())
    assert abs(float(errt.std()) / wantt - 1) < 0.03


def test_per_cell_matches_explicit_cell_model_in_distribution():
    t, r = _reader(ReadNoise(sigma_cell_frac=0.05, adc_bits=None, dac_bits=None))
    x = torch.randn(30, 1, dtype=DTYPE)
    ys = r.mvm(x.repeat(1, 3000))
    g = make_generator(9, "ref")
    sig = 0.05 * float(t.w.abs().max())
    ref = torch.stack(
        [
            (t.w + sig * torch.randn(40, 30, generator=g, dtype=DTYPE)) @ x[:, 0]
            for _ in range(3000)
        ],
        1,
    )
    assert torch.allclose(ys.mean(1), ref.mean(1), atol=0.01)
    assert abs(float(ys.std(1).mean() / ref.std(1).mean()) - 1) < 0.05


def test_output_referred_noise_std_and_mode_gating():
    t, _ = _state()
    x = torch.randn(30, 4000, dtype=DTYPE)
    clean = t.w @ x
    _, r = _reader(
        ReadNoise(
            mode="output_referred",
            sigma_out_frac=0.02,
            sigma_cell_frac=0.5,
            adc_bits=None,
            dac_bits=None,
        )
    )
    err = r.mvm(x) - clean
    want = 0.02 * float(clean.pow(2).mean().sqrt())
    assert abs(float(err.std()) / want - 1) < 0.03  # sigma_cell ignored in this mode
    _, r2 = _reader(
        ReadNoise(
            mode="per_cell", sigma_out_frac=0.5, sigma_cell_frac=0.0, adc_bits=None, dac_bits=None
        )
    )
    assert (r2.mvm(x) - clean).abs().max() < 1e-12  # sigma_out ignored in per_cell mode
    _, r3 = _reader(
        ReadNoise(
            mode="both", sigma_out_frac=0.02, sigma_cell_frac=0.02, adc_bits=None, dac_bits=None
        )
    )
    err3 = r3.mvm(x) - clean
    assert float(err3.std()) > float(err.std())


def test_frozen_persistence():
    nz = ReadNoise(sigma_cell_frac=0.05, persistence="frozen", adc_bits=None, dac_bits=None)
    _, r = _reader(nz)
    x = torch.randn(30, 2, dtype=DTYPE)
    assert torch.equal(r.mvm(x), r.mvm(x))
    assert torch.equal(r.full_readout(), r.full_readout())


def test_one_over_f_path():
    nz = ReadNoise(mode="output_referred", one_over_f=True, adc_bits=None, dac_bits=None)
    t, r = _reader(nz)
    x = torch.randn(30, 3, dtype=DTYPE)
    y = r.mvm(x)
    assert (y - t.w @ x).abs().max() > 0 and torch.isfinite(y).all()


def test_quantization():
    x = torch.linspace(-1, 1, 1001, dtype=DTYPE)
    q = quantize(x, 4)
    assert len(torch.unique(q)) <= 15 and (q - x).abs().max() <= 1 / 7 / 2 + 1e-12
    assert quantize(x, None) is x
    t, r = _reader(ReadNoise(sigma_cell_frac=0.0, adc_bits=6, dac_bits=6))
    x = torch.randn(30, 3, dtype=DTYPE)
    y = r.mvm(x)
    ref = t.w @ x
    assert 0 < (y - ref).abs().max() < 0.1 * ref.abs().max()


def test_state_provider_is_reread_each_call():
    _, st = _state()
    box = {"s": st}
    r = TileReader(lambda: box["s"], QUIET, make_generator(0, "read"))
    x = torch.ones(30, 1, dtype=DTYPE)
    y0 = r.mvm(x)
    box["s"] = st.clone(g_pos=st.g_pos * 1.1, g_neg=st.g_neg * 1.1)
    assert torch.allclose(r.mvm(x), 1.1 * y0)
