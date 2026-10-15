"""Diagonal-Fisher saliency tracker for the co-mitigation controller.

F is an EMA of grad^2 per analog-layer weight: a cheap diagonal-Fisher proxy for
how much the loss moves when that weight is perturbed. The controller ranks stuck
cells by F * (w_stuck - w)^2 - the loss increase retraining would have to absorb
if the cell is left unremapped - so the spare budget goes where it matters.
Whether this beats magnitude/random ordering is a Stage-2 question, not assumed.

Keyed by CrossbarHandle name (f"{module_name}_crossbar_0"), the same keys
system.handles / monitors / pools use, so saliency lookups need no translation.
Memory is one fp32 tensor per layer (26MB for ResNet-9's 6.57M weights);
update() was probed at ~1.2ms, cheap enough for every batch. Call update() AFTER
backward and BEFORE gradient clipping so F sees the raw gradient.
"""

from __future__ import annotations

import torch


class FisherTracker:
    def __init__(self, weights: dict[str, torch.nn.Parameter], decay: float = 0.99):
        if not (0.0 <= decay < 1.0):
            raise ValueError(f"FisherTracker.decay must be in [0, 1), got {decay!r}")
        self.decay = decay
        self._params = dict(weights)
        self.fisher: dict[str, torch.Tensor] = {
            name: torch.zeros_like(p, requires_grad=False) for name, p in self._params.items()
        }
        self.num_updates = 0

    @classmethod
    def from_handles(cls, handles: dict, decay: float = 0.99) -> FisherTracker:
        """One entry per crossbar handle, bound to that handle's layer's weight."""
        return cls({name: h.layer.weight for name, h in handles.items()}, decay)

    @torch.no_grad()
    def update(self) -> None:
        # The first update seeds F with g^2 directly: an EMA started from zeros
        # would under-weight early batches by (1-decay)-sized steps for ~1/(1-decay)
        # updates, making saliency near-zero right after a short training run.
        first = self.num_updates == 0
        for name, p in self._params.items():
            if p.grad is None:
                continue
            g2 = p.grad.detach().pow(2)
            if first:
                self.fisher[name].copy_(g2)
            else:
                self.fisher[name].mul_(self.decay).add_(g2, alpha=1.0 - self.decay)
        self.num_updates += 1

    def state_dict(self) -> dict:
        return {
            "decay": self.decay,
            "num_updates": self.num_updates,
            "fisher": {name: f.detach().cpu().clone() for name, f in self.fisher.items()},
        }

    def load_state_dict(self, state: dict) -> None:
        missing = set(self.fisher) - set(state["fisher"])
        extra = set(state["fisher"]) - set(self.fisher)
        if missing or extra:
            raise ValueError(
                f"FisherTracker state does not match this model's layers "
                f"(missing={sorted(missing)}, unexpected={sorted(extra)})"
            )
        for name, f in state["fisher"].items():
            if f.shape != self.fisher[name].shape:
                raise ValueError(
                    f"FisherTracker state for {name!r} has shape {tuple(f.shape)}, "
                    f"expected {tuple(self.fisher[name].shape)}"
                )
            self.fisher[name].copy_(f.to(self.fisher[name].device))
        self.decay = state["decay"]
        self.num_updates = state["num_updates"]
