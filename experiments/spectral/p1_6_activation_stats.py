# ruff: noqa: C408
"""P1.6: per-layer statistics of real im2col inputs of the trained ResNet-9.

Usage: python experiments/spectral/p1_6_activation_stats.py [--config PATH] [--seeds 42 101]
       [--n-images N] [--allow-dirty]
Caches results/spectral/activations/<seed>/<layer>.pt (stats, PCs, spectra, X subsample) and
writes the run record under results/spectral/p1_6_activations/.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
import torchvision
import yaml

from neurofault.models.cnn import SimpleCNN
from neurofault.spectral.activations import (
    RESNET9_LAYERS,
    capture_im2col_inputs,
    finalize_stats,
    tile_input_norms,
)
from neurofault.spectral.harness import write_run
from neurofault.spectral.tiles import REPO_ROOT, checkpoint_path
from neurofault.spectral.types import make_generator

CONFIG = REPO_ROOT / "experiments" / "configs" / "spectral_p1_6_activations.yaml"


def test_loader(n_images: int, batch_size: int):
    tf = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )  # same as neurofault.data.load_datasets test transform
    ds = torchvision.datasets.CIFAR10(
        root=REPO_ROOT / "data", train=False, download=True, transform=tf
    )
    ds = torch.utils.data.Subset(ds, range(min(n_images, len(ds))))
    return torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=False)


def run_seed(cfg: dict, seed: int) -> pd.DataFrame:
    torch.manual_seed(seed)
    model = SimpleCNN()
    blob = torch.load(checkpoint_path(seed), map_location="cpu", weights_only=False)
    model.load_state_dict(blob["model_state_dict"])
    loader = test_loader(cfg["n_images"], cfg["batch_size"])
    gen = make_generator(seed, "activations")
    raw = capture_im2col_inputs(
        model,
        loader,
        RESNET9_LAYERS,
        cfg["n_images"],
        gen,
        positions_per_image=cfg["positions_per_image"],
        cache_cols=cfg["cache_cols"],
    )
    out_dir = REPO_ROOT / "results" / "spectral" / "activations" / str(seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for wname in RESNET9_LAYERS:
        st = finalize_stats(raw[wname], k=cfg["top_k_pcs"])
        st["tile_input_norm_256"] = tile_input_norms(st["row_energy"], 256)
        st["sampling"] = dict(cfg=cfg, seed=seed, layer=wname)
        torch.save(st, out_dir / f"{wname.removesuffix('.weight')}.pt")
        rows.append(
            dict(
                seed=seed,
                layer=wname,
                n_in=st["n_in"],
                N=st["N"],
                pos_per_img=st["positions_per_image"],
                pr_cov=st["cov"]["participation_ratio"],
                r90_cov=st["cov"]["rank90"],
                r99_cov=st["cov"]["rank99"],
                pr_m2=st["m2"]["participation_ratio"],
                r90_m2=st["m2"]["rank90"],
                r99_m2=st["m2"]["rank99"],
                zero_frac=st["zero_frac"],
                zero_frac_nopad=st["zero_frac_excl_padding"],
                padding_frac=st["padding_frac"],
                mean_energy_share=st["mean_energy_share"],
            )
        )
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=CONFIG)
    ap.add_argument("--seeds", type=int, nargs="*")
    ap.add_argument("--n-images", type=int)
    ap.add_argument("--allow-dirty", action="store_true")
    args = ap.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    if args.n_images:
        cfg["n_images"] = args.n_images
    seeds = args.seeds or cfg["seeds"]
    pd.set_option("display.width", 250)
    for seed in seeds:
        df = run_seed(cfg, seed)
        print(f"\n== seed {seed} ==")
        print(df.drop(columns="seed").to_string(index=False, float_format=lambda v: f"{v:.4g}"))
        write_run(
            REPO_ROOT / "results" / "spectral" / "p1_6_activations",
            cfg,
            seed,
            df,
            params={},
            allow_dirty=args.allow_dirty,
        )


if __name__ == "__main__":
    main()
