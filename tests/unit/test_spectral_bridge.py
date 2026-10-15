"""Bridge tier helpers: F0.8 metric rows, binomial half-width, CrossSim tier-2 write/read."""

from __future__ import annotations

import math

import pytest
import torch

from neurofault.spectral.bridge import (
    abs_dev_over_halfwidth,
    binom_halfwidth_pt,
    f08_metric_rows,
    layer_of_handle,
)


def test_binom_halfwidth_matches_formula():
    hw = binom_halfwidth_pt(90.0, 5000)
    assert hw == pytest.approx(100 * 1.96 * math.sqrt(0.9 * 0.1 / 5000))
    assert 0.8 < binom_halfwidth_pt(89.0, 5000) < 0.9  # ~0.87 pt on the repo's 5000-image test
    assert binom_halfwidth_pt(100.0, 100) > 0  # clamped, never zero


def test_abs_dev_over_halfwidth_scales():
    hw = binom_halfwidth_pt(88.84, 5000)
    assert abs_dev_over_halfwidth(88.84 + hw, 88.84, 5000) == pytest.approx(1.0)
    assert abs_dev_over_halfwidth(88.84, 88.84, 5000) == 0.0


def test_f08_metric_rows_use_frozen_names():
    rows = f08_metric_rows(
        [{"acc_t1": 86.0, "acc_t2": 85.7}],
        [{"acc_ours": 80.0, "acc_aihwkit": 81.5}],
        [{"acc_t1": 88.52, "stored": 88.84}],
        5000,
    )
    assert rows[0]["f08a_tier1_vs_tier2_abs_diff_pt"] == pytest.approx(0.3)
    assert rows[1]["f08b_ours_vs_aihwkit_abs_diff_pt"] == pytest.approx(1.5)
    assert rows[2]["f08c_abs_dev_over_binomial_halfwidth"] == pytest.approx(
        0.32 / binom_halfwidth_pt(88.84, 5000)
    )


def test_layer_of_handle():
    assert layer_of_handle("res1.conv_a_crossbar_0") == "res1.conv_a.weight"
    assert layer_of_handle("fc_crossbar_0") == "fc.weight"


def test_tier_two_roundtrip_orientation_and_independence():
    pytest.importorskip("simulator")  # CrossSim
    from neurofault.spectral.bridge import TierTwo

    class Net(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv2d(3, 4, 3, padding=1)
            self.bn = torch.nn.BatchNorm2d(4)
            self.fc = torch.nn.Linear(4 * 32 * 32, 5)

        def forward(self, x):
            return self.fc(torch.relu(self.bn(self.conv(x))).flatten(1))

    torch.manual_seed(0)
    net = Net().eval()
    # the repo config needs the CIFAR-10 train set for ADC calibration images: skip if absent
    try:
        t2 = TierTwo(net, {"conv": "bn"}, seed=0, n_cal=16)
    except (RuntimeError, FileNotFoundError, OSError) as e:  # pragma: no cover - no dataset
        pytest.skip(f"CIFAR-10 not available: {e}")
    w = net.conv.weight.detach().flatten(1)
    assert torch.allclose(t2.handles["conv.weight"].accessor.read(), w, atol=1e-6)  # orientation
    new = w * 0.7 + 0.01
    t2.set_state({"conv.weight": new}, {"conv.weight": torch.full((4,), 0.5)})
    assert torch.allclose(t2.handles["conv.weight"].accessor.read(), new, atol=1e-6)  # no error
    assert torch.allclose(t2.model.bn.running_mean, net.bn.running_mean - 0.5)
    t2.set_state(None)  # independent calls: back to the checkpoint state
    assert torch.allclose(t2.handles["conv.weight"].accessor.read(), w, atol=1e-6)
    assert torch.allclose(t2.model.bn.running_mean, net.bn.running_mean)
