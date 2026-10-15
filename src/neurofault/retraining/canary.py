"""Held-out canary split of the TRAIN set, for deployment-regime retraining.

Why it exists: after a fault wave the controller needs a cheap, label-backed,
test-set-free signal for "how much did this wave hurt" and "has this burst
plateaued". The real test set must never steer the controller (that would leak
test accuracy into every decision), so the signal comes from `canary_size` train
images that are
  * chosen by this module's OWN torch.Generator(seed) - never the global RNG, so
    building a canary perturbs nothing else (fault maps, burst shuffles) - and
    deterministic for a given (seed, dataset size, canary_size);
  * DISJOINT from the burst training pool (a burst that trained on the canary
    would make canary accuracy a training-set number and the plateau rule
    meaningless);
  * evaluated through the non-augmenting test transform, so the same weights give
    the same canary accuracy every time.
Only built in the deployment regime; the continuous path never calls this and so
consumes no extra RNG.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from neurofault.config import ExperimentConfig
from neurofault.data import load_datasets, load_train_dataset_clean

# Decouples the canary draw from the other seed-derived streams (injection uses
# seed*1000+wave); any fixed offset works, it only needs to be a constant.
_CANARY_SEED_OFFSET = 7_919_000


def canary_sigma(acc_percent: float, n: int) -> float:
    """Standard error (accuracy points) of a canary reading of `acc_percent` on `n` images:
    100 * sqrt(p (1 - p) / n), with p clamped to [0.01, 0.99]."""
    p = min(0.99, max(0.01, acc_percent / 100.0))
    return 100.0 * math.sqrt(p * (1.0 - p) / n)


@dataclass
class CanarySplit:
    canary_indices: list[int]  # sorted, indices into the (shared) train index space
    burst_indices: list[int]  # sorted; everything in the train pool except the canary
    canary_loader: torch.utils.data.DataLoader  # clean transform, no shuffle
    burst_loader: torch.utils.data.DataLoader  # augmenting transform, canary EXCLUDED


def split_indices(pool_size: int, canary_size: int, seed: int) -> tuple[list[int], list[int]]:
    """Pure index split, using a private generator. Raises if the canary would
    leave no burst training data."""
    if canary_size >= pool_size:
        raise ValueError(
            f"canary_size ({canary_size}) must be < the train pool size ({pool_size}) - "
            "the canary is held out of burst training, so some train data must remain"
        )
    g = torch.Generator()
    g.manual_seed(seed + _CANARY_SEED_OFFSET)
    perm = torch.randperm(pool_size, generator=g).tolist()
    return sorted(perm[:canary_size]), sorted(perm[canary_size:])


def build_canary(config: ExperimentConfig) -> CanarySplit:
    train_loader, _ = load_datasets(config)
    # Same train pool the experiment itself draws from: Subset(range(min(train_samples, N))).
    pool_size = len(train_loader.dataset)
    canary_idx, burst_idx = split_indices(pool_size, config.retraining.canary_size, config.seed)

    clean_full = load_train_dataset_clean(config)
    canary_loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(clean_full, canary_idx),
        batch_size=config.batch_size,
        shuffle=False,
    )

    # Burst shuffling gets a private generator too, so iterating it never touches
    # the global RNG's stream (the augmentation transforms still do - that is the
    # burst executor's business, not the split's).
    shuffle_gen = torch.Generator()
    shuffle_gen.manual_seed(config.seed + _CANARY_SEED_OFFSET + 1)
    burst_loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(train_loader.dataset.dataset, burst_idx),
        batch_size=config.batch_size,
        shuffle=True,
        generator=shuffle_gen,
    )
    return CanarySplit(canary_idx, burst_idx, canary_loader, burst_loader)
