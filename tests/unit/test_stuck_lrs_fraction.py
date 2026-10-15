"""stuck_lrs_fraction: share of newly stuck cells pinned to LRS (g_max). Default 0.5 must reproduce the old n // 2 split."""

import pytest
import torch

from neurofault.config import FaultConfig
from neurofault.faults.injection import _custom_stuck_at


class _Acc:
    def __init__(self, m):
        self.m = m

    def read(self):
        return self.m.clone()

    def write(self, m):
        self.m = m.clone()


class _Handle:
    name = "test"

    def __init__(self, m):
        self.accessor = _Acc(m)


class _Mon:
    def __init__(self, shape):
        self.g_min, self.g_max = 0.0, 1.0
        self.stuck_mask = torch.zeros(shape, dtype=torch.bool)
        self.last_injected_mask = torch.zeros(shape, dtype=torch.bool)
        self.health_scores = torch.ones(shape)
        self.stability_index = torch.ones(shape)
        self.stress = torch.zeros(shape)


def _run(frac=None, seed=0):
    m = torch.full((40, 50), 0.5)
    h, mon = _Handle(m), _Mon(m.shape)
    kw = {} if frac is None else {"stuck_lrs_fraction": frac}
    cfg = FaultConfig(density=0.2, **kw)
    g = torch.Generator().manual_seed(seed)
    n = _custom_stuck_at(h, mon, cfg, g)
    out = h.accessor.m
    return n, int((out == mon.g_max).sum()), int((out == mon.g_min).sum()), out


def test_default_equals_explicit_half_and_old_split():
    n, hi, lo, out = _run(None)
    n2, hi2, lo2, out2 = _run(0.5)
    assert n == n2 == 400 and (hi, lo) == (hi2, lo2) == (200, 200)
    assert torch.equal(out, out2)


@pytest.mark.parametrize(
    "frac,expect_hi", [(0.0, 0), (1.0, 400), (0.1622, 400 - int(400 * (1 - 0.1622)))]
)
def test_fraction_sets_lrs_share(frac, expect_hi):
    n, hi, lo, _ = _run(frac)
    assert hi == expect_hi and hi + lo == n


def test_out_of_range_rejected():
    with pytest.raises(ValueError):
        FaultConfig(stuck_lrs_fraction=1.5)
