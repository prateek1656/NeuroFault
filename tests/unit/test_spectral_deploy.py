"""H5 deployable M4: rank truncation of the multiplicative cosine correction (methods_a)."""

from __future__ import annotations

import torch

from neurofault.spectral.methods_a import (
    FieldFit,
    _cos_factors,
    deployable_correction,
    project_cosine_coeffs,
    rank_factors,
)
from neurofault.spectral.types import DTYPE, make_generator

M, N, ORDER = 24, 20, 4


def _field_from_coeffs(c):
    ca, cb = _cos_factors(M, N, ORDER)
    return ca.T @ c @ cb


def test_projection_roundtrip():
    c = torch.randn(ORDER, ORDER, generator=make_generator(0, "c"), dtype=DTYPE)
    assert torch.allclose(project_cosine_coeffs(_field_from_coeffs(c), ORDER), c, atol=1e-10)


def test_rank1_exact_for_planted_rank1_field():
    g = make_generator(1, "u")
    u, v = (
        torch.randn(ORDER, generator=g, dtype=DTYPE),
        torch.randn(ORDER, generator=g, dtype=DTYPE),
    )
    u[0] = v[0] = 1.0
    field = _field_from_coeffs(torch.outer(u, v))
    a, b = rank_factors(project_cosine_coeffs(field, ORDER), (M, N), 1)
    assert a.shape[1] == 1
    assert torch.allclose(a @ b.T, field, atol=1e-10)


def test_rank2_needs_two_terms():
    g = make_generator(2, "u")
    c = torch.randn(ORDER, ORDER, generator=g, dtype=DTYPE)
    c = c[:, :1] @ c[:1, :] + c[:, 1:2] @ c[1:2, :]  # exact rank 2
    field = _field_from_coeffs(c)
    cc = project_cosine_coeffs(field, ORDER)
    a1, b1 = rank_factors(cc, (M, N), 1)
    a2, b2 = rank_factors(cc, (M, N), 2)
    assert (a1 @ b1.T - field).abs().max() > 1e-3
    assert torch.allclose(a2 @ b2.T, field, atol=1e-10) and a2.shape[1] == 2


def test_deployable_recovers_w0_and_counts_passes():
    g = make_generator(3, "w")
    w0 = torch.randn(M, N, generator=g, dtype=DTYPE)
    u = torch.tensor([-0.4, 0.1, 0.05, 0.0], dtype=DTYPE)
    v = torch.tensor([1.0, 0.2, 0.0, 0.1], dtype=DTYPE)
    f = _field_from_coeffs(torch.outer(u, v))  # planted rank-1 F
    wt = w0 * (1 + f)
    w_full, terms = deployable_correction(wt, f, ORDER, None)
    assert terms >= 1
    # G = 1/(1+F) is not exactly in the span, but the full-rank projection is close to W0
    assert (w_full - w0).norm() / w0.norm() < 0.05
    _, t1 = deployable_correction(wt, f, ORDER, 1)
    _, t2 = deployable_correction(wt, f, ORDER, 2)
    assert t1 == 1 and t2 == 2  # extra passes = terms - 1 = 0 and 1
    # a purely multiplicative scalar field is corrected exactly by rank 1
    w1, t = deployable_correction(w0 * 0.5, torch.full((M, N), -0.5, dtype=DTYPE), ORDER, 1)
    assert t == 1 and torch.allclose(w1, w0, atol=1e-10)


def test_fieldfit_deploy_option_matches_function():
    from neurofault.spectral.readout import TileReader
    from neurofault.spectral.types import ConductanceState, EncodingSpec, ReadNoise

    g = make_generator(4, "w")
    w0 = torch.randn(M, N, generator=g, dtype=DTYPE)
    f = _field_from_coeffs(torch.tensor([[-0.3, 0.1, 0, 0]] + [[0.0] * 4] * 3, dtype=DTYPE))
    cur = [w0.clone()]
    rd = TileReader(
        lambda: ConductanceState(cur[0].clone(), None, 1.0, EncodingSpec("signed_ideal")),
        ReadNoise("per_cell", 0.0, adc_bits=None, dac_bits=None),
        make_generator(4, "r"),
    )
    ff = FieldFit("cos", order=ORDER, p_fwd=8, p_bwd=8, seed=1, deploy=True, deploy_rank=1)
    ff.bind(lambda: cur[0], lambda: w0)
    ff.calibrate(rd)
    cur[0] = w0 * (1 + f)
    c = ff.update(rd, 1.0)
    want, _ = deployable_correction(cur[0], c.meta["F"], ORDER, 1)
    assert torch.allclose(c.W_eff, want) and c.meta["extra_passes"] == 0
    assert torch.allclose(c.meta["F"], f, atol=1e-5)
