"""Plain, backend-unaware CNN for CIFAR-10. Which layers become memristive
is decided entirely by neurofault.crossbar.array.patch_layers - this module
knows nothing about crossbars, so it stays importable and unit-testable
without any particular backend installed at all.
"""

from __future__ import annotations

import torch
from torch import nn


def _conv_bn_relu(in_channels: int, out_channels: int) -> nn.Sequential:
    block = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )
    # Zero-init each block's BN gamma is deliberately NOT done here - unlike
    # _ResidualBlock below, these are plain feature-extraction stages, not
    # residual branches, so there's no "should start near-identity" argument.
    return block


class _ResidualPair(nn.Module):
    """Two conv-bn-relu stages, added back onto their own input (identity
    shortcut, channels unchanged) - the res1/res2 blocks in the canonical
    ResNet-9 (David Page/Myrtle.ai "How to Train Your ResNet", the same
    architecture IBM's real 92.81%-CIFAR-10 chip paper used - Le Gallo et
    al. 2023, Nature Electronics, verified via the arXiv preprint's Methods
    section, not guessed). Channels never change here, so no projection
    shortcut is needed, unlike neurofault's prior _ResidualBlock."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv_a = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn_a = nn.BatchNorm2d(channels)
        self.conv_b = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn_b = nn.BatchNorm2d(channels)
        # Same zero-init rationale as before (Bag of Tricks, He et al. 2018):
        # the residual branch starts as a no-op, so each block starts as a
        # clean identity map.
        nn.init.zeros_(self.bn_b.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.relu(self.bn_a(self.conv_a(x)))
        out = self.bn_b(self.conv_b(out))
        return torch.relu(x + out)


class SimpleCNN(nn.Module):
    """ResNet-9 (David Page/Myrtle.ai's canonical CIFAR-10 architecture,
    64/128/256/512 channels - not IBM's core-size-constrained 56/112/224
    rescaling of the same design, since the standard/most-reproduced
    version aids comparability more than replicating a chip-specific detail
    that only exists because of their crossbar core dimensions).

    Replaces the prior 3-residual-block architecture entirely (superseding the earlier widened-only and plain-deep variants, which regressed).
    That prior architecture, even after the plain-deep regression was fixed
    with skip connections, plateaued at 49.5%/56.24% final/max on the real
    full 50,000-sample CIFAR-10 run (GPU-verified) - well
    below any defensible ceiling. Root cause, verified by reading IBM's
    actual paper (Le Gallo et al. 2023) rather than assumed: their own
    ResNet-9 has FEWER parameters than the architecture it replaces here
    (1.87M vs ~3.04M) yet reaches 93.67% in pure digital training - the
    problem was never capacity, it was that this project had never once
    run a training recipe matching any real published CIFAR-10 result
    (Adam instead of SGD, 30 epochs instead of IBM's 300+200, batch 32
    instead of 128, no Cutout). This architecture change is step one of
    replicating that recipe faithfully; see experiments/run.py and
    neurofault.data for the optimizer/schedule/augmentation half of it.
    """

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.prep = _conv_bn_relu(3, 64)

        self.layer1 = nn.Sequential(_conv_bn_relu(64, 128), nn.MaxPool2d(2))
        self.res1 = _ResidualPair(128)

        self.layer2 = nn.Sequential(_conv_bn_relu(128, 256), nn.MaxPool2d(2))
        self.layer3 = nn.Sequential(_conv_bn_relu(256, 512), nn.MaxPool2d(2))
        self.res2 = _ResidualPair(512)

        self.pool = nn.MaxPool2d(4)
        self.fc = nn.Linear(512, num_classes)
        # A well-documented DAWNBench/"How to Train Your ResNet" detail:
        # scaling the final logits down stabilizes early training under a
        # high one-cycle peak LR - kept as a plain float multiply, not a
        # learnable parameter, so it doesn't add a crossbar-eligible layer.
        self.logit_scale = 0.125

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.prep(x)
        x = self.layer1(x)
        x = self.res1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.res2(x)
        x = self.pool(x)

        # reshape(), not view(): AIHWKit's AnalogConv2d can return a
        # non-contiguous tensor, which view() rejects.
        x = x.reshape(x.size(0), -1)
        return self.fc(x) * self.logit_scale


class TinyCNN(nn.Module):
    """~42K-weight CIFAR-shaped stand-in for SimpleCNN, used by the fast test path
    (model_arch="tiny_cnn"). Same module conventions as SimpleCNN - bias-free
    conv-bn-relu stages, a _ResidualPair, a bias-ful final Linear, a plain-float
    logit_scale - and the same attribute names it shares (prep/layer1/res1/layer2/
    pool/fc), so crossbar patching, handle naming, FisherTracker, BN recalibration
    and checkpoint load treat it exactly like the real network. It is a logic-test
    vehicle only: no accuracy claim is ever made about it."""

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.prep = _conv_bn_relu(3, 16)
        self.layer1 = nn.Sequential(_conv_bn_relu(16, 32), nn.MaxPool2d(2))
        self.res1 = _ResidualPair(32)
        self.layer2 = nn.Sequential(_conv_bn_relu(32, 64), nn.MaxPool2d(2))
        self.pool = nn.MaxPool2d(8)
        self.fc = nn.Linear(64, num_classes)
        self.logit_scale = 0.125

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.prep(x)
        x = self.layer1(x)
        x = self.res1(x)
        x = self.layer2(x)
        x = self.pool(x)
        x = x.reshape(x.size(0), -1)
        return self.fc(x) * self.logit_scale


class VGG6(nn.Module):
    """Plain VGG-style 6-conv network (64-64-128-128-256-256, ~1.15M weights), the SECOND
    network for the net2 study (model_arch="vgg6"). Differs from ResNet-9 in block type (no
    residual pairs), depth/width (6 convs, max 256 channels, fewer weights) and layout.
    Same conventions as SimpleCNN: bias-free conv-bn-relu, one bias-ful Linear named `fc`,
    plain-float logit_scale (not crossbar-eligible)."""

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            _conv_bn_relu(3, 64),
            _conv_bn_relu(64, 64),
            nn.MaxPool2d(2),
            _conv_bn_relu(64, 128),
            _conv_bn_relu(128, 128),
            nn.MaxPool2d(2),
            _conv_bn_relu(128, 256),
            _conv_bn_relu(256, 256),
            nn.MaxPool2d(2),
        )
        self.pool = nn.MaxPool2d(4)
        self.fc = nn.Linear(256, num_classes)
        self.logit_scale = 0.125

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(self.features(x))
        x = x.reshape(x.size(0), -1)
        return self.fc(x) * self.logit_scale


class _ViTAttention(nn.Module):
    """Multi-head self-attention. The qkv and output projections are plain nn.Linear (they map
    to crossbars). The score matmul (q k^T), the softmax and the attention-weighted sum stay
    digital: they multiply two activations, not an activation and a stored weight. This is a
    scope choice of the net3 study."""

    def __init__(self, dim: int, heads: int):
        super().__init__()
        assert dim % heads == 0
        self.heads = heads
        self.scale = (dim // heads) ** -0.5
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, d = x.shape
        # Linear layers get 2-D input (b*n, d): CrossSim's AnalogLinear backward gives a wrongly
        # shaped bias gradient for 3-D input, and 2-D is the path every other net uses.
        qkv = self.qkv(x.reshape(b * n, d)).reshape(b, n, 3, self.heads, d // self.heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)
        attn = torch.softmax((q @ k.transpose(-2, -1)) * self.scale, dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(b * n, d)
        return self.proj(out).reshape(b, n, d)


class _ViTBlock(nn.Module):
    """Pre-norm transformer block: LayerNorm -> attention -> add, LayerNorm -> MLP -> add."""

    def __init__(self, dim: int, heads: int, mlp_ratio: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = _ViTAttention(dim, heads)
        self.norm2 = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, dim * mlp_ratio)
        self.fc2 = nn.Linear(dim * mlp_ratio, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, d = x.shape
        x = x + self.attn(self.norm1(x))
        h = self.fc2(torch.nn.functional.gelu(self.fc1(self.norm2(x).reshape(b * n, d))))
        return x + h.reshape(b, n, d)


class ViTTiny(nn.Module):
    """Small ViT for CIFAR (model_arch="vit_tiny"), the THIRD network of the study: patch size 4
    (64 tokens), dim 192, 4 pre-norm blocks, 4 heads, MLP ratio 2, mean-pool head, ~1.2M weights.
    No BatchNorm: LayerNorm is digital and has no running statistics, so BN recalibration is a
    no-op for this net (recalibrate_bn returns 0). Crossbar layers: patch embedding (a Linear on
    flattened 4x4x3 patches, so every mapped layer is nn.Linear), qkv, proj, fc1, fc2 in each
    block, and the head `fc`. Digital: attention scores, softmax, GELU, LayerNorm, position
    embedding, residual adds."""

    def __init__(
        self,
        num_classes: int = 10,
        dim: int = 192,
        depth: int = 4,
        heads: int = 4,
        mlp_ratio: int = 2,
        patch: int = 4,
        image: int = 32,
    ):
        super().__init__()
        self.patch = patch
        n_tokens = (image // patch) ** 2
        self.embed = nn.Linear(3 * patch * patch, dim)
        self.pos = nn.Parameter(torch.zeros(1, n_tokens, dim))
        nn.init.trunc_normal_(self.pos, std=0.02)
        self.blocks = nn.Sequential(*[_ViTBlock(dim, heads, mlp_ratio) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
        self.fc = nn.Linear(dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        p = self.patch
        # (b, c, h/p, p, w/p, p) -> (b, h/p * w/p, c*p*p)
        x = x.reshape(b, c, h // p, p, w // p, p).permute(0, 2, 4, 1, 3, 5)
        x = x.reshape(b * (h // p) * (w // p), c * p * p)
        x = self.embed(x).reshape(b, (h // p) * (w // p), -1) + self.pos
        x = self.norm(self.blocks(x)).mean(dim=1)
        return self.fc(x)


_MODEL_ARCHS = {"resnet9": SimpleCNN, "tiny_cnn": TinyCNN, "vgg6": VGG6, "vit_tiny": ViTTiny}


def build_model(config) -> nn.Module:
    """The single construction point for the network (pretrain, system build, predeploy,
    deployment/checkpoint load, calibration sweep, tests). Dispatches on
    config.model_arch."""
    return _MODEL_ARCHS[config.model_arch](num_classes=getattr(config, "num_classes", 10))
