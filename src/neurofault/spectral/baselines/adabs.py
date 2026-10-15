"""B3 AdaBS (Joshi 2020): refit digital BatchNorm statistics on a few calibration images.

WHY a thin wrapper: AdaBS is a network-level fix (digital BN stats), so it has no tile-level
W_eff and is excluded from tile metrics. Here we only fix the interface and the
cost accounting: images forwarded (ReadCost.images) and the MVM-equivalent
images x analog_layers_tiles. The network bridge calls ``apply``; the analog matrices are left
bit-identical (retraining.bn_recal contract).
"""

from __future__ import annotations

import math

import torch

from neurofault.retraining.bn_recal import recalibrate_bn
from neurofault.spectral.types import ReadCost


class AdaBS:
    def __init__(self, n_images: int = 256, batch_size: int = 256):
        self.n_images, self.batch_size = n_images, batch_size
        self.name = f"B3_adabs_{n_images}"

    @property
    def n_batches(self) -> int:
        return math.ceil(self.n_images / self.batch_size)

    def planned_cost(self) -> ReadCost:
        return ReadCost(images=self.n_batches * self.batch_size)

    def mvm_equivalents(self, n_tiles: int) -> int:
        """Each calibration image drives every analog tile once (forward)."""
        return self.planned_cost().images * n_tiles

    def apply(self, model: torch.nn.Module, loader, device: torch.device | str) -> ReadCost:
        used = recalibrate_bn(model, loader, self.n_batches, device)
        return ReadCost(images=used * self.batch_size)
