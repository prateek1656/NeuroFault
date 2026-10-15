# ruff: noqa: C408
"""Real layer-input (im2col) statistics of the trained ResNet-9 .

Why: Direction A's M3 weights corrections by the input covariance and its rel_output_error needs
held-out X; Direction B's concurrent detection (E5) needs the rank/sparsity of real activations.

Convention: a conv layer's matrix is W = weight.flatten(1) with shape
(C_out, C_in*kh*kw); the matching input column of an output position is
``F.unfold(x, k, padding, stride)[b, :, l]`` (channel-major, then kh, then kw), so
``W @ X`` reproduces ``conv(x)`` exactly (verified in tests). Tiles are blocks
W[r0:r0+m, c0:c0+n], so the tile sees input rows X[c0:c0+n] (`slice_tile_inputs`).
X is (n_in, N): one column per (image, spatial position) sample. fc layers see the pooled
512-vector, one column per image.

Memory: the full X of res2 (4608 x 16 positions x 5000 images) is ~1.5 GB, so statistics are
accumulated in a streaming float64 second-moment matrix over a random subset of
``positions_per_image`` positions per image (all of them if the layer has fewer), and only a
small random column subsample (``cache_cols``) of X is kept for held-out error evaluation.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch import nn

from neurofault.spectral.types import DTYPE, TileSpec

# input of each weight layer is whatever the module receives, so the hook sits on the module
RESNET9_LAYERS: dict[str, str] = {
    "prep.0.weight": "prep.0",
    "layer1.0.0.weight": "layer1.0.0",
    "res1.conv_a.weight": "res1.conv_a",
    "res1.conv_b.weight": "res1.conv_b",
    "layer2.0.0.weight": "layer2.0.0",
    "layer3.0.0.weight": "layer3.0.0",
    "res2.conv_a.weight": "res2.conv_a",
    "res2.conv_b.weight": "res2.conv_b",
    "fc.weight": "fc",
}


def im2col(x: torch.Tensor, module: nn.Module) -> tuple[torch.Tensor, torch.Tensor]:
    """(B, n_in, L) input columns and a same-shaped bool mask (False = padding zero).

    Linear layers give (B, n_in, 1). Column order == weight.flatten(1) column order."""
    if isinstance(module, nn.Linear):
        cols = x.reshape(x.shape[0], -1, 1)
        return cols, torch.ones_like(cols, dtype=torch.bool)
    if not isinstance(module, nn.Conv2d):
        raise TypeError(f"unsupported layer type {type(module).__name__}")
    if module.groups != 1:
        raise ValueError("grouped conv not supported")
    kw = dict(
        kernel_size=module.kernel_size,
        padding=module.padding,
        stride=module.stride,
        dilation=module.dilation,
    )
    cols = F.unfold(x, **kw)
    valid = F.unfold(torch.ones(1, *x.shape[1:], dtype=x.dtype, device=x.device), **kw) > 0
    return cols, valid.expand_as(cols)


def slice_tile_inputs(X: torch.Tensor, spec: TileSpec) -> torch.Tensor:
    """Rows of X seen by the tile `spec`: X[c0:c0+n, :]."""
    return X[spec.c0 : spec.c0 + spec.n]


def tile_input_norms(row_energy: torch.Tensor, tile_size: int) -> torch.Tensor:
    """sqrt(E ||x_block||^2) for consecutive column blocks of `tile_size` (last may be shorter).

    row_energy[i] = E[x_i^2] (stored as `row_energy` in the stats)."""
    n = row_energy.numel()
    return torch.stack([row_energy[c : c + tile_size].sum().sqrt() for c in range(0, n, tile_size)])


@dataclass
class _Acc:
    n_in: int
    G: torch.Tensor  # sum x x^T (n,n) float64
    s: torch.Tensor  # sum x (n,)
    N: int = 0
    n_zero: int = 0  # exact zeros among all entries (incl. padding)
    n_zero_valid: int = 0  # exact zeros at non-padding entries
    n_valid: int = 0
    n_entries: int = 0
    cache: list[torch.Tensor] = field(default_factory=list)
    positions_total: int = 0
    positions_used: int = 0


@torch.no_grad()
def capture_im2col_inputs(
    model: nn.Module,
    loader: Sequence | torch.utils.data.DataLoader,
    layers: dict[str, str] | Sequence[str],
    n_images: int,
    gen: torch.Generator,
    *,
    positions_per_image: int = 16,
    cache_cols: int = 4096,
    device: str | torch.device = "cpu",
) -> dict[str, dict]:
    """Run `n_images` images (eval mode) and accumulate per-layer input statistics.

    layers: {state_dict weight name: module name} or just weight names (".weight" stripped).
    Returns {weight_name: acc} with raw accumulators converted by `finalize_stats`-ready dicts:
    keys G, s, N, zero counts, cache X (n_in, <=cache_cols float32), sampling record."""
    if not isinstance(layers, dict):
        layers = {w: w.removesuffix(".weight") for w in layers}
    modules = dict(model.named_modules())
    model.eval().to(device)
    accs: dict[str, _Acc] = {}
    seen = {"n": 0, "batch": 0}
    hooks = []

    def make_hook(wname: str):
        def hook(mod, args):
            x = args[0].detach().to(DTYPE)
            cols, valid = im2col(x, mod)  # (B, n, L)
            B, n, L = cols.shape
            P = min(positions_per_image, L)
            if P < L:
                idx = torch.stack([torch.randperm(L, generator=gen)[:P] for _ in range(B)])
                idx = idx.to(cols.device)
                gi = idx[:, None, :].expand(B, n, P)
                cols, valid = cols.gather(2, gi), valid.gather(2, gi)
            S = cols.permute(1, 0, 2).reshape(n, B * P)
            V = valid.permute(1, 0, 2).reshape(n, B * P)
            a = accs.get(wname)
            if a is None:
                a = accs[wname] = _Acc(
                    n, torch.zeros(n, n, dtype=DTYPE), torch.zeros(n, dtype=DTYPE)
                )
            a.G += (S @ S.T).cpu()
            a.s += S.sum(1).cpu()
            a.N += S.shape[1]
            z = S == 0
            a.n_zero += int(z.sum())
            a.n_zero_valid += int((z & V).sum())
            a.n_valid += int(V.sum())
            a.n_entries += S.numel()
            a.positions_total += B * L
            a.positions_used += B * P
            k = min(S.shape[1], max(1, -(-cache_cols * B // n_images)))
            pick = torch.randperm(S.shape[1], generator=gen)[:k].to(S.device)
            a.cache.append(S[:, pick].float().cpu())

        return hook

    for wname, mname in layers.items():
        hooks.append(modules[mname].register_forward_pre_hook(make_hook(wname)))
    try:
        for batch in loader:
            xb = batch[0]
            take = min(xb.shape[0], n_images - seen["n"])
            if take <= 0:
                break
            model(xb[:take].to(device))
            seen["n"] += take
    finally:
        for h in hooks:
            h.remove()
    out = {}
    for wname, a in accs.items():
        X = torch.cat(a.cache, 1)[:, :cache_cols]
        out[wname] = dict(
            n_in=a.n_in,
            N=a.N,
            G=a.G,
            s=a.s,
            X_cache=X,
            n_images=seen["n"],
            zero_frac=a.n_zero / a.n_entries,
            zero_frac_excl_padding=a.n_zero_valid / max(a.n_valid, 1),
            padding_frac=1 - a.n_valid / a.n_entries,
            positions_total=a.positions_total,
            positions_used=a.positions_used,
            positions_per_image=min(positions_per_image, a.positions_total // max(seen["n"], 1)),
        )
    return out


def _energy_rank(evals: torch.Tensor, frac: float) -> int:
    cum = torch.cumsum(evals, 0) / evals.sum()
    return int(torch.searchsorted(cum, torch.tensor(frac, dtype=cum.dtype) - 1e-12).item()) + 1


def _spectrum_stats(M: torch.Tensor, k: int) -> dict:
    ev, V = torch.linalg.eigh(M)
    ev = ev.flip(0).clamp_min(0)
    V = V.flip(1)
    tot = ev.sum()
    return dict(
        eigvals=ev,
        pcs=V[:, :k].float(),
        participation_ratio=float(tot**2 / (ev**2).sum()),
        rank90=_energy_rank(ev, 0.90),
        rank99=_energy_rank(ev, 0.99),
        total=float(tot),
    )


def finalize_stats(raw: dict, *, k: int = 64) -> dict:
    """Mean, covariance + second-moment spectra, ranks, zero/padding fractions, row energies.

    Two spectra: `cov` (centred, textbook PCs) and `m2` (uncentred second moment E[xx^T], the
    quantity that weights ||(W^ - W) x||^2 and so what M3/rel_output_error actually).
    Ranks: participation ratio (sum l)^2/sum l^2 and smallest r holding 90%/99% of the energy."""
    N = raw["N"]
    mu = raw["s"] / N
    M2 = raw["G"] / N
    cov = M2 - torch.outer(mu, mu)
    cov = (cov + cov.T) / 2
    k = min(k, raw["n_in"])
    tot_m2 = float(torch.diagonal(M2).sum())
    return dict(
        n_in=raw["n_in"],
        N=N,
        n_images=raw["n_images"],
        mean=mu,
        row_energy=torch.diagonal(M2).clone(),
        mean_energy_share=float(mu @ mu) / tot_m2 if tot_m2 > 0 else 0.0,
        zero_frac=raw["zero_frac"],
        zero_frac_excl_padding=raw["zero_frac_excl_padding"],
        padding_frac=raw["padding_frac"],
        positions_per_image=raw["positions_per_image"],
        positions_total=raw["positions_total"],
        positions_used=raw["positions_used"],
        cov=_spectrum_stats(cov, k),
        m2=_spectrum_stats(M2, k),
        X_cache=raw["X_cache"],
    )
