# ruff: noqa: C408
"""im2col order exactness and stats on synthetic data."""

from __future__ import annotations

import torch
from torch import nn

from neurofault.spectral.activations import (
    capture_im2col_inputs,
    finalize_stats,
    im2col,
    slice_tile_inputs,
    tile_input_norms,
)
from neurofault.spectral.types import TileSpec, make_generator


class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv2d(3, 5, 3, padding=1, bias=False)
        self.c2 = nn.Conv2d(5, 4, 3, padding=1, stride=2, bias=False)
        self.fc = nn.Linear(4 * 4 * 4, 3, bias=False)

    def forward(self, x):
        x = torch.relu(self.c1(x))
        x = torch.relu(self.c2(x))
        return self.fc(x.flatten(1))


def test_im2col_matches_conv_exactly():
    torch.manual_seed(0)
    m = Tiny().double()
    x = torch.randn(3, 3, 8, 8, dtype=torch.float64)
    cols, valid = im2col(x, m.c1)
    W = m.c1.weight.flatten(1)
    y = torch.einsum("on,bnl->bol", W, cols).reshape(3, 5, 8, 8)
    assert torch.allclose(y, m.c1(x), atol=1e-12)
    assert not valid.all()  # padding positions flagged
    cols2, _ = im2col(torch.randn(2, 5, 8, 8, dtype=torch.float64), m.c2)
    assert cols2.shape == (2, 45, 16)  # stride 2


def test_capture_stats_and_tile_slicing():
    torch.manual_seed(1)
    m = Tiny()
    xs = torch.randn(20, 3, 8, 8)
    loader = [(xs[i : i + 10], torch.zeros(10)) for i in (0, 10)]
    gen = make_generator(0, "act")
    raw = capture_im2col_inputs(
        m,
        loader,
        ["c1.weight", "c2.weight", "fc.weight"],
        20,
        gen,
        positions_per_image=8,
        cache_cols=50,
    )
    assert raw["c1.weight"]["N"] == 20 * 8 and raw["fc.weight"]["N"] == 20
    assert raw["c1.weight"]["X_cache"].shape[0] == 27
    st = finalize_stats(raw["c2.weight"], k=8)
    assert st["n_in"] == 45 and st["cov"]["pcs"].shape == (45, 8)
    # full-population check for the fc layer (no position subsampling)
    with torch.no_grad():
        h = torch.relu(m.c2(torch.relu(m.c1(xs)))).flatten(1).T.double()
    sf = finalize_stats(raw["fc.weight"], k=4)
    assert torch.allclose(sf["mean"], h.mean(1), atol=1e-6)
    assert torch.allclose(sf["row_energy"], (h**2).mean(1), atol=1e-6)
    spec = TileSpec(0, "c2.weight", 0, 10, 4, 20, 20)
    X = raw["c2.weight"]["X_cache"]
    assert slice_tile_inputs(X, spec).shape == (20, X.shape[1])
    assert tile_input_norms(sf["row_energy"], 16).shape == (4,)


def test_rank_zero_and_mean_share_on_synthetic():
    g = torch.Generator().manual_seed(0)
    n, N, r = 40, 5000, 5
    basis = torch.linalg.qr(torch.randn(n, r, generator=g, dtype=torch.float64))[0]
    X = basis @ torch.randn(r, N, generator=g, dtype=torch.float64) + 2.0
    X[:, :1000] = torch.where(torch.rand(n, 1000, generator=g) < 0.5, 0.0, X[:, :1000].double())
    raw = dict(
        n_in=n,
        N=N,
        G=X @ X.T,
        s=X.sum(1),
        X_cache=X[:, :10].float(),
        n_images=N,
        zero_frac=float((X == 0).double().mean()),
        zero_frac_excl_padding=0.0,
        padding_frac=0.0,
        positions_total=N,
        positions_used=N,
        positions_per_image=1,
    )
    st = finalize_stats(raw, k=5)
    assert 0.0 < st["zero_frac"] < 0.2
    assert st["mean_energy_share"] > 0.3
    # exactly-low-rank data without zeros
    X2 = basis @ torch.randn(r, N, generator=g, dtype=torch.float64)
    st2 = finalize_stats(dict(raw, G=X2 @ X2.T, s=X2.sum(1)), k=5)
    assert st2["cov"]["rank99"] <= r and st2["cov"]["participation_ratio"] <= r + 1e-6
    assert st2["m2"]["rank90"] <= r
    assert st2["mean_energy_share"] < 0.01
