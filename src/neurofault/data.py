"""Dataset loading, ported from Implementation.ipynb cell 5 - the only code
in this repo proven to have actually loaded and trained on real data -
parameterized via ExperimentConfig instead of the notebook's hardcoded
range(2000)/range(1000).
"""

from __future__ import annotations

import torch
import torchvision

from neurofault.config import ExperimentConfig


def load_train_dataset_clean(config: ExperimentConfig):
    """The same TRAIN split load_datasets() draws from, but through the
    non-augmenting test-time transform. Used to build a canary evaluation set
    (retraining/canary.py): canary accuracy must be a deterministic function of
    the weights, which RandomCrop/Flip/Cutout would make it not be. Constructing
    it consumes no RNG, so the continuous path is unaffected. Returns the FULL
    train set (indices are shared with load_datasets()'s Subset(range(n)))."""
    if config.dataset in ("cifar10", "cifar100"):
        cls = (
            torchvision.datasets.CIFAR100
            if config.dataset == "cifar100"
            else torchvision.datasets.CIFAR10
        )
        transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
        return cls(root="./data", train=True, download=True, transform=transform)
    if config.dataset == "mnist":
        transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.Grayscale(num_output_channels=3),
                torchvision.transforms.Resize((32, 32)),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
        return torchvision.datasets.MNIST(
            root="./data", train=True, download=True, transform=transform
        )
    raise ValueError(f"Unknown dataset: {config.dataset!r}")


def load_datasets(
    config: ExperimentConfig,
) -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    if config.dataset in ("cifar10", "cifar100"):
        # CIFAR-100 uses the identical recipe and the same (0.5,0.5,0.5) normalization as
        # CIFAR-10 (existing convention; no dataset-specific mean/std).
        # Real gap found during development: with no augmentation at all, the model plateaued by
        # epoch
        # 2-3 regardless of epoch count or how much more data it was given -
        # it converges onto the exact same fixed set of images fast and
        # stops improving, since there's no variation left to learn from.
        # Standard CIFAR-10 recipe (random crop with padding + horizontal
        # flip) - train-only, never applied to the test set, which must stay
        # clean/deterministic for a fair, reproducible accuracy measurement.
        train_transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.RandomCrop(32, padding=4),
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                # Cutout (DeVries & Taylor 2017), 4x4 patch - matches IBM's
                # own published ResNet-9 recipe exactly (Le Gallo et al. 2023, verified against the
                # arXiv preprint's Methods section). torchvision has no literal
                # "Cutout" transform; RandomErasing with a fixed square
                # scale/ratio and p=1.0 (always applied, matching the
                # original Cutout paper - not RandomErasing's own default
                # p=0.5) is the same operation. scale=(0.0156, 0.0156) is
                # exactly 4x4 pixels on a 32x32 image (16/1024).
                torchvision.transforms.RandomErasing(
                    p=1.0, scale=(0.0156, 0.0156), ratio=(1.0, 1.0), value=0
                ),
            ]
        )
        test_transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
        cls = (
            torchvision.datasets.CIFAR100
            if config.dataset == "cifar100"
            else torchvision.datasets.CIFAR10
        )
        train_full = cls(root="./data", train=True, download=True, transform=train_transform)
        test_full = cls(root="./data", train=False, download=True, transform=test_transform)

    elif config.dataset == "mnist":
        transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.Grayscale(num_output_channels=3),
                torchvision.transforms.Resize((32, 32)),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
        train_full = torchvision.datasets.MNIST(
            root="./data", train=True, download=True, transform=transform
        )
        test_full = torchvision.datasets.MNIST(
            root="./data", train=False, download=True, transform=transform
        )

    else:
        raise ValueError(f"Unknown dataset: {config.dataset!r}")

    train_subset = torch.utils.data.Subset(
        train_full, range(min(config.train_samples, len(train_full)))
    )
    test_subset = torch.utils.data.Subset(
        test_full, range(min(config.test_samples, len(test_full)))
    )

    trainloader = torch.utils.data.DataLoader(
        train_subset, batch_size=config.batch_size, shuffle=True
    )
    testloader = torch.utils.data.DataLoader(
        test_subset, batch_size=config.batch_size, shuffle=False
    )

    return trainloader, testloader
