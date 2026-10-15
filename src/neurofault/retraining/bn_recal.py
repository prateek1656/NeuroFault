"""Forward-only BatchNorm running-statistic recalibration.

After a fault wave the analog layers' output statistics shift, so the digital
nn.BatchNorm2d running mean/var (fitted to the healthy network) are stale; refitting
them costs a few forward passes and no gradient step. Verified before building on it:
BN layers are plain digital nn.BatchNorm2d (unpatched by the crossbar backends), and
this procedure leaves every analog matrix bit-identical.

Procedure: reset_running_stats(), momentum=None (cumulative average over the
recalibration batches, not an EMA that would keep a trace of the stale stats), ONLY
the BN modules in train mode (everything else stays eval - model.train() globally would
also flip analog layers' / dropout behavior), no_grad forwards. Momentum and every
module's train/eval flag are restored afterwards. No optimizer, no synchronize().
"""

from __future__ import annotations

import torch
from torch.nn.modules.batchnorm import _BatchNorm


def recalibrate_bn(
    model: torch.nn.Module, loader, n_batches: int, device: torch.device | str
) -> int:
    """Refit BN running stats over up to `n_batches` batches of `loader`. Returns the
    number of batches actually used (0 = model has no BN layer or n_batches == 0, in
    which case nothing is touched)."""
    bns = [m for m in model.modules() if isinstance(m, _BatchNorm)]
    if not bns or n_batches <= 0:
        return 0

    saved_training = {m: m.training for m in model.modules()}
    saved_momentum = {bn: bn.momentum for bn in bns}
    used = 0
    try:
        model.eval()
        for bn in bns:
            bn.reset_running_stats()
            bn.momentum = None
            bn.train()
        with torch.no_grad():
            for inputs, _targets in loader:
                if used >= n_batches:
                    break
                model(inputs.to(device))
                used += 1
    finally:
        for bn, momentum in saved_momentum.items():
            bn.momentum = momentum
        for m, flag in saved_training.items():
            m.training = flag
    return used
