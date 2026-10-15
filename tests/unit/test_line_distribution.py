"""distribution="line" (row/column fault segments): count, bounds, shape, determinism."""

import torch

from neurofault.config import FaultConfig
from neurofault.faults.injection import _pick_indices


def _pick(n, seed=0, rows=256, cols=1152):
    g = torch.Generator().manual_seed(seed)
    return _pick_indices(rows, cols, n, "line", torch.device("cpu"), g)


def test_line_count_close_and_unique_and_in_bounds():
    rows, cols = 256, 1152
    for density in (0.01, 0.05, 0.10):
        n = int(rows * cols * density)
        idx = _pick(n)
        assert idx.numel() == idx.unique().numel()
        assert int(idx.max()) < rows * cols and int(idx.min()) >= 0
        assert idx.numel() >= 0.97 * n


def test_line_faults_are_collinear_segments():
    rows, cols = 256, 1152
    idx = _pick(2000)
    mask = torch.zeros(rows, cols, dtype=torch.bool)
    mask.view(-1)[idx] = True
    # most faulty cells have a faulty horizontal or vertical neighbour (random would not)
    nb = torch.zeros_like(mask)
    nb[:, 1:] |= mask[:, :-1]
    nb[:, :-1] |= mask[:, 1:]
    nb[1:, :] |= mask[:-1, :]
    nb[:-1, :] |= mask[1:, :]
    assert (mask & nb).sum() / mask.sum() > 0.7


def test_line_is_deterministic_and_config_accepts_it():
    assert torch.equal(_pick(500, seed=3), _pick(500, seed=3))
    FaultConfig(distribution="line")
