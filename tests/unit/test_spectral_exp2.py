import numpy as np
import torch

from neurofault.spectral.exp2 import (
    adabs_images_per_tile,
    adabs_mvm_equivalents,
    cluster_bootstrap_ci,
    fold_conv_bn,
    pareto_front,
    split_batches,
)


def test_fold_matches_bn_eval():
    torch.manual_seed(0)
    conv = torch.nn.Conv2d(3, 5, 3, padding=1, bias=False)
    bn = torch.nn.BatchNorm2d(5)
    bn.running_mean.normal_()
    bn.running_var.uniform_(0.5, 2.0)
    bn.weight.data.normal_()
    bn.bias.data.normal_()
    bn.eval()
    x = torch.randn(2, 3, 8, 8)
    w, c = fold_conv_bn(conv.weight, bn.weight, bn.bias, bn.running_mean, bn.running_var, bn.eps)
    ref = bn(conv(x))
    out = torch.nn.functional.conv2d(x, w, padding=1) + c.reshape(1, -1, 1, 1)
    assert torch.allclose(ref, out, atol=1e-5)


def test_adabs_cost():
    assert adabs_images_per_tile([10, 4], [2, 3]) == 32
    assert adabs_mvm_equivalents(256, 32) == 8192


def test_split_batches():
    assert split_batches(2600) == [260] * 10
    assert split_batches(64) == [64]
    s = split_batches(1000)
    assert sum(s) == 1000 and max(s) <= 260 and max(s) - min(s) <= 1


def test_bootstrap_ci_covers_mean_and_sign():
    v = np.array([1.0, 1.2, 0.8, 1.1, 0.9, 1.0])
    c = np.array([0, 0, 1, 1, 2, 2])
    m, lo, hi = cluster_bootstrap_ci(v, c, n_boot=500)
    assert lo <= m <= hi and lo > 0


def test_pareto_front():
    pts = [(1, 0.5), (2, 0.4), (3, 0.9), (4, 0.9), (0.5, 0.1)]
    assert pareto_front(pts) == [4, 0, 2]
