"""Direction-A methods (M4 field fit, M3, M2): exact noise-free recovery and counted costs."""

from __future__ import annotations

import torch

from neurofault.spectral.methods_a import (
    FieldFit,
    InputSubspace,
    ResidualSketch,
    cosine_basis,
)
from neurofault.spectral.readout import TileReader
from neurofault.spectral.types import DTYPE, ReadNoise, make_generator

NOISELESS = ReadNoise("per_cell", 0.0, adc_bits=None, dac_bits=None)


def _setup(f: torch.Tensor, m=24, n=20, seed=0):
    g = make_generator(seed, "t")
    w0 = torch.randn(m, n, generator=g, dtype=DTYPE)
    cur = [w0]

    class _S:  # minimal state provider with weights()
        shape = (m, n)

        def weights(self):
            return cur[0]

    from neurofault.spectral.types import ConductanceState, EncodingSpec

    def prov():
        return ConductanceState(cur[0].clone(), None, 1.0, EncodingSpec("signed_ideal"))

    rd = TileReader(prov, NOISELESS, make_generator(seed, "r"))
    return w0, cur, rd


def _run(method, w0, cur, rd, f):
    method.bind(lambda: cur[0], lambda: w0)
    method.calibrate(rd)
    cur[0] = w0 * (1 + f)
    return method.update(rd, 1.0)


def test_cos_basis_recovers_smooth_field_exactly():
    m, n = 24, 20
    basis = cosine_basis(m, n, 3)
    c = torch.randn(9, generator=make_generator(1, "c"), dtype=DTYPE) * 0.05
    f = torch.einsum("b,bij->ij", c, basis)
    w0, cur, rd = _setup(f, m, n)
    corr = _run(FieldFit("cos", order=3, p_fwd=4, p_bwd=4), w0, cur, rd, f)
    assert torch.allclose(corr.meta["F"], f, atol=1e-6)
    assert torch.allclose(corr.W_eff, w0, atol=1e-6)
    assert corr.cost.mvm_fwd == 4 and corr.cost.mvm_bwd == 4


def test_rank1_and_rank2_planted_field_recovered():
    m, n = 24, 20
    gen = make_generator(2, "f")
    for r in (1, 2):
        u = torch.randn(m, r, generator=gen, dtype=DTYPE) * 0.1
        v = torch.randn(n, r, generator=gen, dtype=DTYPE)
        f = u @ v.T
        w0, cur, rd = _setup(f, m, n, seed=r)
        corr = _run(FieldFit("rank", rank=r, p_fwd=6, p_bwd=6, iters=80), w0, cur, rd, f)
        err = (corr.meta["F"] - f).norm() / f.norm()
        assert err < 1e-5, (r, float(err))


def test_forward_only_rank1():
    m, n = 24, 20
    gen = make_generator(3, "f")
    f = torch.outer(torch.randn(m, generator=gen, dtype=DTYPE) * 0.1, torch.ones(n, dtype=DTYPE))
    w0, cur, rd = _setup(f, m, n)
    corr = _run(FieldFit("rank", rank=1, p_fwd=6, p_bwd=0, iters=30), w0, cur, rd, f)
    assert (corr.meta["F"] - f).norm() / f.norm() < 1e-5
    assert corr.cost.mvm_bwd == 0


def test_m3_exact_on_subspace_and_cost():
    m, n = 16, 12
    w0, cur, rd = _setup(None, m, n)
    u, _ = torch.linalg.qr(torch.randn(n, 3, generator=make_generator(4, "u"), dtype=DTYPE))
    m3 = InputSubspace(u)
    m3.bind(lambda: cur[0], lambda: w0)
    m3.calibrate(rd)
    cur[0] = w0 * 1.3
    corr = m3.update(rd, 1.0)
    assert torch.allclose(corr.W_eff @ u, w0 @ u, atol=1e-9)
    assert corr.cost.mvm_fwd == 3


def test_m2_sketch_recovers_lowrank_residual():
    m, n = 20, 16
    w0, cur, rd = _setup(None, m, n)
    sk = ResidualSketch(k=2, s=4)
    sk.calibrate(rd)
    gen = make_generator(5, "e")
    e = torch.randn(m, 2, generator=gen, dtype=DTYPE) @ torch.randn(
        2, n, generator=gen, dtype=DTYPE
    )
    cur[0] = w0 + e
    r_k, cost = sk.update(rd, 1.0)
    assert torch.allclose(r_k, e, atol=1e-8)
    assert cost.mvm_fwd == 6 and cost.mvm_bwd == 6
