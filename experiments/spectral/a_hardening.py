# ruff: noqa: C408
"""Phase H / H2: M4 against stronger calibration baselines on BN-folded deployment (a_hardening).

Registered in criteria/a_hardening.yaml (frozen 2026-10-10, committed before any run). Platform = a_bnfold
(W2 BN-folded ResNet-9, CIFAR-10 and CIFAR-100, ckpt seeds 42/101/102, scenarios D2 / D3 {hotspot, rank_r,
row_gradient} x rho {1, 2} / D4, t in {1h, 1d, 30d, 1yr}, 2% per-cell probe read noise, 10k-image eval) with
ALL random streams offset +2000. Methods per (dataset, seed, scenario, t):
  M4      strict FieldFit(cos, 4), 8 fwd + 7 bwd probes + 1 mean-bias read = 16 MVMs/tile
  B8      per-tile best of GDC/row/col/bi + mean-bias (selected by held-out error, optimistic incumbent)
  HB1_N8 / HB1_N32   digital per-channel affine recal from N augmented TRAIN images (hardening.py)
  HB2     4 interleaved reference lines per tile, 15 probe MVMs + 1 mean-bias read (hardening.py)
  HB3     rank-4 timer-switched adapters trained offline on the nominal drift model (hardening.py)
  NONE    uncompensated drifted network (context)
Deviations from a_bnfold, all deliberate and documented in the run's notes:
- the drift field of every scenario is sampled on an AUGMENTED tile (m array rows + 4 reference lines
  interleaved), because HB2's reference lines must sit in the same die field; the array part of that draw is
  what ALL methods see (so they stay paired). Statistically equivalent to a_bnfold's draw, not bit-identical.
- network evaluation runs on the Apple GPU (MPS, float32) when available (--eval-device cpu to override);
  same forward for every method.
- calibration images (HB1) and adapter training activations (HB3) are AUGMENTED TRAIN images (no test
  leakage; CIFAR-100 train is data/cifar100_hf/train.npz, fetched from HF uoft-cs/cifar100 like the test set).
  B8's held-out selection error still uses the cached test-image im2col columns, exactly as in a_bnfold.

  python experiments/spectral/a_hardening.py --jobs 3 --allow-dirty [--smoke]
"""

from __future__ import annotations

import copy
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import a_common as ac
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from a_bnfold import BFBridge, _tiles, act_dir
from a_common import BN_OF, CONV_LAYERS, ENC, INCUMBENTS, PROG, TileCtx, best_of, make_drift
from a_exp2_jobs import POSITIONS, _mk_get_tile
from f0_common import REPO_ROOT, finish, load_config, parse_args

from neurofault.spectral.baselines import GDC, BiScale, PerColumn, PerRow, bind_all
from neurofault.spectral.drift import PCMPerDevice
from neurofault.spectral.encoding import encode
from neurofault.spectral.hardening import (
    HB3_GRID_S,
    RefColumnAffine,
    TimerLowRankAdapters,
    hb1_calibrate,
    hb1_mvm_equivalents,
    interleave_index,
    merge_rows,
    reference_weights,
    take_rows,
)
from neurofault.spectral.harness import run_grid
from neurofault.spectral.methods_a import FieldFit
from neurofault.spectral.noise import program
from neurofault.spectral.readout import TileReader
from neurofault.spectral.types import DTYPE, ReadNoise, Tile, make_generator

SEED_OFFSET = 2000
POOL_SEED_BASE = 7000 + SEED_OFFSET  # a_exp2 used 7000 + ckpt_seed
HB1_POOL = 32  # HB1 images are the first N of this pool; HB3 trains on the images after it
BASELINES = ("HB1_N8", "HB1_N32", "HB2", "HB3")
METHODS = ("M4", "B8", "HB1_N8", "HB1_N32", "HB2", "HB3", "NONE")
_ACT: dict = {}
_EVAL_DEVICE = os.environ.get("A_HARDENING_DEVICE", "mps")


def cache_dir(out: Path) -> Path:
    return out / "_cache"


# ------------------------------------------------------------------------------- data


def train_pool(dataset: str, seed: int, n: int) -> torch.Tensor:
    """n AUGMENTED train images (crop pad 4 + flip + 4x4 Cutout, the checkpoints' training
    augmentation; a_exp2_jobs.Exp2Bridge recipe), Normalize(0.5, 0.5); fixed per (dataset, seed)."""
    import torchvision

    T = torchvision.transforms
    aug = T.Compose([
        T.RandomCrop(32, padding=4), T.RandomHorizontalFlip(), T.ToTensor(),
        T.Normalize((0.5,) * 3, (0.5,) * 3),
        T.RandomErasing(p=1.0, scale=(0.0156, 0.0156), ratio=(1.0, 1.0), value=0),
    ])  # fmt: skip
    g = torch.Generator().manual_seed(POOL_SEED_BASE + seed)
    if dataset == "cifar10":
        ds = torchvision.datasets.CIFAR10(
            REPO_ROOT / "data", train=True, download=False, transform=aug
        )
        get = lambda i: ds[i][0]
        total = len(ds)
    else:
        from PIL import Image

        z = np.load(REPO_ROOT / "data" / "cifar100_hf" / "train.npz")
        xs = z["x"]
        get = lambda i: aug(Image.fromarray(xs[i]))
        total = len(xs)
    idx = torch.randperm(total, generator=g)[:n].tolist()
    torch.manual_seed(POOL_SEED_BASE + seed)  # augmentation draws
    return torch.stack([get(i) for i in idx])


class HBridge(BFBridge):
    """BFBridge with evaluation on `device` and a CPU copy of the model for calibration passes."""

    def __init__(self, dataset: str, ckpt_seed: int, n_eval: int, device: str):
        super().__init__(dataset, ckpt_seed, n_eval)
        self.cpu_model = copy.deepcopy(self.model).eval()
        self.dev = device if (device != "mps" or torch.backends.mps.is_available()) else "cpu"
        self.model.to(self.dev)
        self.xe, self.ye = self.xe.to(self.dev), self.ye.to(self.dev)

    @torch.no_grad()
    def acc(self, mats, bias=None, ada_n: int = 0) -> float:
        self.model.load_state_dict(self.orig)
        sd = self.model.state_dict()
        for name, w in (mats or {}).items():
            sd[name].copy_(w.reshape(sd[name].shape).to(sd[name]))
            if bias and name in bias:
                sd[f"{BN_OF[name.removesuffix('.weight')]}.running_mean"].sub_(
                    bias[name].to(sd[name])
                )
        self.model.eval()
        correct = 0
        for i in range(0, len(self.xe), 500):
            correct += int(
                (self.model(self.xe[i : i + 500]).argmax(1) == self.ye[i : i + 500]).sum()
            )
        return 100.0 * correct / len(self.xe)


# ------------------------------------------------------------------------------- platform patching


def _install(dataset: str, ckpt_seed: int, bridge: BFBridge):
    saved = (ac.get_tile, ac.act_stats, ac.make_generator)
    ac.get_tile = _mk_get_tile(bridge)

    def act_stats(seed, layer):
        k = (dataset, seed, layer)
        if k not in _ACT:
            p = act_dir(dataset, seed) / f"{layer.removesuffix('.weight')}.pt"
            _ACT[k] = torch.load(p, weights_only=False)
        return _ACT[k]

    ac.act_stats = act_stats
    ac.make_generator = lambda seed, stream, *a, **k: make_generator(
        seed + SEED_OFFSET, stream, *a, **k
    )
    return saved


def _restore(saved):
    ac.get_tile, ac.act_stats, ac.make_generator = saved


# ------------------------------------------------------------------------------- HB3 offline inputs


def cov_path(out: Path, dataset: str, seed: int) -> Path:
    return cache_dir(out) / f"cov_{dataset}_{seed}.pt"


def cov_job(p: dict) -> list[dict]:
    """Second-moment matrices E[x x^T] of every (layer, column block) im2col input of the DIGITAL
    folded network on augmented TRAIN images (images HB1_POOL..HB1_POOL+train_images of the pool)."""
    torch.set_num_threads(1)
    cfg, seed, ds, out = p["cfg"], p["seed"], p["dataset"], Path(p["out"])
    h3 = cfg["hb3"]
    t0 = time.time()
    bridge = BFBridge(ds, seed, 10)
    model = bridge.model.eval()
    imgs = train_pool(ds, seed, HB1_POOL + h3["train_images"])[HB1_POOL:]
    gen = make_generator(seed + SEED_OFFSET, "hb3_cov_positions")
    blocks: dict[tuple[str, int], torch.Tensor] = {}
    count = {"n": 0}
    cur: dict = {}

    def mk_hook(layer):
        def hook(_m, inp):
            x = F.unfold(inp[0], 3, padding=1)  # (B, n, HW)
            b, n, hw = x.shape
            idx = torch.randint(0, hw, (b, h3["positions_per_image"]), generator=gen)
            xs = torch.gather(x, 2, idx[:, None, :].expand(b, n, idx.shape[1]))
            xs = xs.permute(1, 0, 2).reshape(n, -1).to(DTYPE)
            for c0 in range(0, n, 256):
                blk = xs[c0 : c0 + 256]
                blocks[(layer, c0)] = blocks.get((layer, c0), 0) + blk @ blk.T
            cur["cols"] = xs.shape[1]

        return hook

    hs = [model.get_submodule(L.removesuffix(".weight")).register_forward_pre_hook(mk_hook(L))
          for L in CONV_LAYERS]  # fmt: skip
    with torch.no_grad():
        for i in range(0, len(imgs), 64):
            model(imgs[i : i + 64])
            count["n"] += cur["cols"]
    for h in hs:
        h.remove()
    total = count["n"]
    cov = {k: (v / total).float() for k, v in blocks.items()}
    cov_path(out, ds, seed).parent.mkdir(parents=True, exist_ok=True)
    tmp = cov_path(out, ds, seed).with_suffix(".tmp")
    torch.save(dict(cov=cov, n_samples=total), tmp)
    tmp.replace(cov_path(out, ds, seed))
    return [dict(dataset=ds, ckpt_seed=seed, cov_samples=total, cov_wall_s=time.time() - t0)]


# ------------------------------------------------------------------------------- per-tile engine


class HRun:
    """All methods on one tile: incumbents + strict M4 (own noise streams, as a_bnfold), HB2 on the
    augmented tile, HB3 adapters. Every estimator reads through a counted TileReader."""

    def __init__(
        self, ctx: TileCtx, real_aug, aug0, geo, sigma, rseed, hb2_cfg, hb3: TimerLowRankAdapters
    ):
        self.ctx, self.real_aug = ctx, real_aug
        self.arr_idx, self.ref_idx, ref_pos = geo
        self.cur, self.cur_aug = [ctx.stp], [aug0]
        noise = ReadNoise("per_cell", sigma, adc_bits=None, dac_bits=None)
        mk = lambda tag, prov: TileReader(prov, noise, make_generator(rseed, f"{tag}:{ctx.key}"))
        self.reader = mk("read", lambda: self.cur[0])
        self.reader2 = mk("read2", lambda: self.cur[0])  # M4
        self.reader3 = mk("read3", lambda: self.cur_aug[0])  # HB2 (augmented tile)
        sd = rseed + (sum(ord(ch) for ch in repr(ctx.key)) & 0xFFFF)
        w_now, w0 = (lambda: self.cur[0].weights()), (lambda: ctx.tile.w)
        self.methods = {
            "gdc": GDC("ls", p=8, seed=sd),
            "row": PerRow("rademacher", 8, "ls", seed=sd),
            "col": PerColumn("rademacher", 8, "backward", seed=sd),
            "bi": BiScale(8, 8, seed=sd),
        }
        bind_all(list(self.methods.values()), w_now, w0)
        for m in self.methods.values():
            m.calibrate(self.reader)
        self.y0bar = self.reader.mvm(ctx.xbar[:, None])[:, 0]
        self.m4 = FieldFit("cos", order=4, p_fwd=8, p_bwd=7, seed=sd)
        bind_all([self.m4], w_now, w0)
        self.m4.calibrate(self.reader2)
        self.hb2 = RefColumnAffine(
            self.ref_idx, self.arr_idx, ref_pos, p=hb2_cfg["p_probes"], seed=sd
        )
        bind_all([self.hb2], w_now, w0)
        self.hb2.calibrate(self.reader3)
        self.y0bar3 = self.reader3.mvm(ctx.xbar[:, None])[self.arr_idx, 0]
        self.hb3 = hb3
        bind_all([self.hb3], w_now, w0)

    def _with_bias(self, c, xread, wt, y0bar):
        ctx = self.ctx
        b = y0bar - (xread + (c.W_eff - wt) @ ctx.xbar)
        return dict(W=c.W_eff, b=b, meta=c.meta, cost=c.cost)

    def at(self, t: float) -> dict:
        ctx = self.ctx
        aug = self.real_aug.at(t)
        self.cur_aug[0], self.cur[0] = aug, take_rows(aug, self.arr_idx)
        wt = self.cur[0].weights()
        xread = self.reader.mvm(ctx.xbar[:, None])[:, 0]
        out = {"_wt": wt}
        for k, m in self.methods.items():
            r = self._with_bias(m.update(self.reader, t), xread, wt, self.y0bar)
            r["err"] = ctx.err(r["W"], r["b"])
            out[k] = r
        c = self.m4.update(self.reader2, t)
        out["m4"] = self._with_bias(c, self.reader2.mvm(ctx.xbar[:, None])[:, 0], wt, self.y0bar)
        c = self.hb2.update(self.reader3, t)
        x3 = self.reader3.mvm(ctx.xbar[:, None])[self.arr_idx, 0]
        out["hb2"] = self._with_bias(c, x3, wt, self.y0bar3)
        c = self.hb3.update(self.reader, t)
        out["hb3"] = dict(W=c.W_eff, b=None, meta=c.meta, cost=c.cost)
        return out


def _nominal_dbar(ctx: TileCtx, grid, mc: int, rseed: int):
    """Nominal-model (programming noise + D2 drift, MC mean) per grid time: the LS scalar gain
    alpha_g = <W0, E W> / <E W, E W> and the residual E[W0] - alpha_g E[W] the adapter must carry."""
    nom = PCMPerDevice(drift_scale=1.0)
    acc = [torch.zeros_like(ctx.tile.w) for _ in grid]
    for s in range(mc):
        gen = make_generator(rseed, f"hb3nom:{ctx.key}:{s}")
        real = nom.sample(program(ctx.st0, PROG, gen), gen)
        for i, g in enumerate(grid):
            acc[i] += real.at(g).weights()
    w0, ew = ctx.tile.w, [a / mc for a in acc]
    alphas = [float((w0 * e).sum() / (e * e).sum()) for e in ew]
    return alphas, [w0 - a * e for a, e in zip(alphas, ew, strict=True)]


# ------------------------------------------------------------------------------- HB1 network pass


class _Stop(Exception):
    pass


def hb1_network(
    bridge: HBridge,
    w_none: dict,
    tiles_by_layer: dict,
    imgs: torch.Tensor,
    refs: dict,
    sigma: float,
    gen: torch.Generator,
):
    """Sequential per-channel affine recal of the 8 analog layers from `imgs` (CPU float32). Analog
    outputs are W(t) x plus the same per-cell read noise as the probe reads (per tile, per position)."""
    model = bridge.cpu_model
    n_img = len(imgs)

    def analog(layer, corr):
        model.load_state_dict(bridge.orig)
        sd = model.state_dict()
        for L in CONV_LAYERS:  # drifted weights everywhere; earlier layers carry their correction
            w = w_none[L].reshape(sd[L].shape).float()
            if L in corr:
                g, b = corr[L]
                w = w * g.float().reshape(-1, 1, 1, 1)
                sd[f"{BN_OF[L.removesuffix('.weight')]}.bias"].add_(b.float())
            sd[L].copy_(w)
        box = {}

        def hook(_m, inp):
            box["x"] = inp[0]
            raise _Stop

        h = model.get_submodule(layer.removesuffix(".weight")).register_forward_pre_hook(hook)
        try:
            with torch.no_grad():
                model(imgs)
        except _Stop:
            pass
        finally:
            h.remove()
        x = box["x"]
        wl = w_none[layer].float()
        xu = F.unfold(x, 3, padding=1)  # (B, n, HW)
        y = torch.einsum("mn,bnp->mbp", wl, xu)  # (C, B, HW) analog output, noise-free part
        if sigma > 0:
            for r0, c0, m, n in tiles_by_layer[layer]:
                wmax = float(wl[r0 : r0 + m, c0 : c0 + n].abs().max())
                nrm = xu[:, c0 : c0 + n, :].norm(dim=1)  # (B, HW)
                y[r0 : r0 + m] += (
                    sigma * wmax * nrm[None] * torch.randn(m, *nrm.shape, generator=gen)
                )
        return y.reshape(y.shape[0], -1)

    def ref(layer):
        return refs[layer][:, :n_img].reshape(refs[layer].shape[0], -1)

    return hb1_calibrate(CONV_LAYERS, analog, ref)


def digital_refs(bridge: HBridge, imgs: torch.Tensor) -> dict:
    """Stored digital reference: pre-bias conv outputs (C, B, HW) of the DIGITAL folded network."""
    model = bridge.cpu_model
    model.load_state_dict(bridge.orig)
    refs, hs = {}, []
    for L in CONV_LAYERS:

        def hook(_m, _i, o, L=L):
            refs[L] = o.detach().permute(1, 0, 2, 3).reshape(o.shape[1], o.shape[0], -1).float()

        hs.append(model.get_submodule(L.removesuffix(".weight")).register_forward_hook(hook))
    with torch.no_grad():
        model(imgs)
    for h in hs:
        h.remove()
    return refs


# ------------------------------------------------------------------------------- jobs


def oracle_job(p: dict) -> list[dict]:
    torch.set_num_threads(1)
    cfg, seed, ds = p["cfg"], p["seed"], p["dataset"]
    bridge = HBridge(ds, seed, cfg["n_eval"], p["device"])
    saved = _install(ds, seed, bridge)
    try:
        mats = bridge.layer_mats()
        pm = {}
        for layer, r0, c0, m, n in _tiles(bridge):
            ctx = TileCtx(seed, layer, r0, c0, m, n, 256)
            pm.setdefault(layer, torch.zeros_like(mats[layer], dtype=torch.float32))
            pm[layer][r0 : r0 + m, c0 : c0 + n] = ctx.wp.float()
        return [dict(dataset=ds, ckpt_seed=seed, acc_digital=bridge.acc(None), acc_prog=bridge.acc(pm),
                     n_eval=cfg["n_eval"], n_tiles=len(_tiles(bridge)))]  # fmt: skip
    finally:
        _restore(saved)


def net_job(p: dict) -> list[dict]:
    torch.set_num_threads(1)
    cfg, seed, ds, sc, out = p["cfg"], p["seed"], p["dataset"], p["scenario"], Path(p["out"])
    h2, h3 = cfg["hb2"], cfg["hb3"]
    bridge = HBridge(ds, seed, cfg["n_eval"], p["device"])
    saved = _install(ds, seed, bridge)
    rseed = seed + SEED_OFFSET
    try:
        mats = bridge.layer_mats()
        mod = make_drift(sc["model"], sc.get("field") or "row_gradient", sc["rho"], 1.0)
        cov = torch.load(cov_path(out, ds, seed), weights_only=False)["cov"]
        tiles = _tiles(bridge)
        tiles_by_layer: dict[str, list] = {}
        for layer, r0, c0, m, n in tiles:
            tiles_by_layer.setdefault(layer, []).append((r0, c0, m, n))
        grid, k_ref = HB3_GRID_S, h2["k_ref"]
        runs, hb3_s, hb3_params = [], 0.0, 0
        for layer, r0, c0, m, n in tiles:
            ctx = TileCtx(seed, layer, r0, c0, m, n, 256)
            arr_idx, ref_idx, ref_pos = interleave_index(m, k_ref)
            absmax = float(ctx.tile.w.abs().max())
            wref = reference_weights(
                ctx.tile.w, k_ref, n, make_generator(rseed, f"hb2ref:{ctx.key}")
            )
            ref0 = encode(Tile(ctx.tile.spec, wref), ENC, absmax=absmax)
            assert abs(ref0.scale - ctx.stp.scale) < 1e-9 * ctx.stp.scale
            ref_stp = program(ref0, PROG, make_generator(rseed, f"hb2prog:{ctx.key}"))
            aug0 = merge_rows(ctx.stp, ref_stp, arr_idx, ref_idx)
            real_aug = mod.sample(aug0, make_generator(rseed, f"drift:{layer}:{r0}:{c0}"))
            t_fit = time.time()
            al, db = _nominal_dbar(ctx, grid, h3["mc_samples"], rseed)
            hb3 = TimerLowRankAdapters(grid, rank=h3["rank"]).fit(
                db, cov[(layer, c0)].to(DTYPE), al
            )
            hb3_s += time.time() - t_fit
            hb3_params += hb3.n_params()
            runs.append(((layer, r0, c0, m, n), HRun(ctx.slim(), real_aug, aug0, (arr_idx, ref_idx, ref_pos),
                                                     cfg["sigma"], rseed, h2, hb3)))  # fmt: skip
        n_t = len(runs)
        per_img = sum(POSITIONS[L] * len(tiles_by_layer[L]) for L in CONV_LAYERS)
        pool = train_pool(ds, seed, HB1_POOL)
        refs = digital_refs(bridge, pool)
        area = (
            k_ref * sum(t[4] for t in tiles) / sum(t[3] * t[4] for t in tiles)
        )  # extra lines / array lines
        rows = []
        for tlabel, t in cfg["times"].items():
            names = ("none", "b8", "m4", "hb2", "hb3")
            W = {k: {} for k in names}
            B = {k: {} for k in names}
            mvm = dict.fromkeys(names, 0)
            for (layer, r0, c0, m, n), run in runs:
                res = run.at(t)
                for name in names:
                    key = best_of(res, INCUMBENTS) if name == "b8" else name
                    r = res[key] if key != "none" else None
                    w = res["_wt"] if r is None else r["W"]
                    W[name].setdefault(layer, torch.zeros_like(mats[layer], dtype=torch.float32))
                    W[name][layer][r0 : r0 + m, c0 : c0 + n] = w.float()
                    if r is not None and r["b"] is not None:
                        bd = B[name].setdefault(layer, torch.zeros(mats[layer].shape[0]))
                        bd[r0 : r0 + m] += r["b"].float()
                        mvm[name] += 1  # the layer-mean-input bias read (counted for every method)
                    if r is not None and r.get("cost") is not None:
                        mvm[name] += int(r["cost"].mvm_fwd + r["cost"].mvm_bwd)
            acc = {k: bridge.acc(W[k], B[k] or None) for k in names}
            cost = {
                "M4": mvm["m4"],
                "B8": mvm["b8"],
                "HB2": mvm["hb2"],
                "HB3": mvm["hb3"],
                "NONE": 0,
            }
            for n_img in cfg["hb1_ns"]:
                gen = make_generator(rseed, f"hb1noise:{tlabel}:{n_img}")
                corr = hb1_network(bridge, W["none"], tiles_by_layer, pool[:n_img], refs,
                                   cfg["sigma"], gen)  # fmt: skip
                mh = {L: (corr[L][0][:, None] * W["none"][L].double()).float() for L in CONV_LAYERS}
                bh = {L: corr[L][1].float() for L in CONV_LAYERS}
                acc[f"hb1_{n_img}"] = bridge.acc(mh, bh)
                cost[f"HB1_N{n_img}"] = hb1_mvm_equivalents(n_img, per_img)
            base = dict(dataset=ds, ckpt_seed=seed, scenario=sc["name"], model=sc["model"],
                        field=sc.get("field") or "", rho=sc["rho"], time=tlabel, t_s=t)  # fmt: skip
            accs = {"M4": acc["m4"], "B8": acc["b8"], "HB2": acc["hb2"], "HB3": acc["hb3"],
                    "NONE": acc["none"], **{f"HB1_N{n}": acc[f"hb1_{n}"] for n in cfg["hb1_ns"]}}  # fmt: skip
            for meth in METHODS:
                rows.append(dict(**base, method=meth, acc=accs[meth], cost_mvm_eq=cost[meth],
                                 n_tiles=n_t, area_overhead_frac=area if meth == "HB2" else 0.0,
                                 hb3_fit_s=hb3_s, hb3_params=hb3_params, n_eval=cfg["n_eval"]))  # fmt: skip
            del W, B
        return rows
    finally:
        _restore(saved)


# ------------------------------------------------------------------------------- metrics


def build_metrics(net: pd.DataFrame, orc: pd.DataFrame) -> pd.DataFrame:
    key = ["dataset", "ckpt_seed", "scenario", "t_s"]
    m4 = net[net.method == "M4"].set_index(key)[["acc", "cost_mvm_eq"]].rename(
        columns={"acc": "acc_m4", "cost_mvm_eq": "cost_m4"})  # fmt: skip
    df = net.join(m4, on=key).merge(orc[["dataset", "ckpt_seed", "acc_prog", "acc_digital"]],
                                    on=["dataset", "ckpt_seed"], how="left")  # fmt: skip
    for b in BASELINES:
        is_b = df.method == b
        df[f"m4_minus_{b}_pt"] = np.where(is_b, df.acc_m4 - df.acc, np.nan)
        df[f"{b}_cost_le_m4"] = np.where(is_b, (df.cost_mvm_eq <= df.cost_m4).astype(float), np.nan)
    return df


def wide_table(df: pd.DataFrame) -> pd.DataFrame:
    """dataset x t_s mean accuracy per method (+ mean cost, and programmed accuracy)."""
    g = df.groupby(["dataset", "t_s", "method"]).agg(
        acc=("acc", "mean"), cost=("cost_mvm_eq", "mean")
    )
    return g.reset_index()


def main(argv=None) -> int:
    ap_args = list(sys.argv[1:] if argv is None else argv)
    device = _EVAL_DEVICE
    if "--eval-device" in ap_args:
        i = ap_args.index("--eval-device")
        device = ap_args[i + 1]
        del ap_args[i : i + 2]
    a = parse_args(__doc__.splitlines()[0], "a_hardening.yaml", "a_hardening", ap_args)
    cfg = load_config(a.config, a.smoke)
    cfg.update(master_seed=a.seed, eval_device=device, hb3_grid_s=list(HB3_GRID_S))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    from a_bnfold import ensure_c100_stats

    if "cifar100" in cfg["datasets"]:
        ensure_c100_stats(cfg["ckpt_seeds"])
    ocfg = {k: v for k, v in cfg.items() if k != "scenarios"}
    opts = [dict(cfg=ocfg, seed=s, dataset=d, out=str(out), device=device)
            for d in cfg["datasets"] for s in cfg["ckpt_seeds"]]  # fmt: skip
    pts = [dict(cfg=cfg, seed=s, dataset=d, scenario=sc, out=str(out), device=device)
           for sc in cfg["scenarios"] for d in cfg["datasets"] for s in cfg["ckpt_seeds"]]  # fmt: skip
    t0 = time.time()
    cov = run_grid(cov_job, opts, jobs=a.jobs, out_dir=out / "_cells_cov")
    orc = run_grid(oracle_job, opts, jobs=a.jobs, out_dir=out / "_cells_oracle")
    net = run_grid(net_job, pts, jobs=a.jobs, out_dir=out / "_cells_net")
    df = build_metrics(net, orc)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    print(
        wide_table(df)
        .pivot_table(index=["dataset", "t_s"], columns="method", values="acc")
        .round(2)
    )
    _run_dir, _ = finish("A-hardening H2", "a_hardening.yaml", cfg, a, df,
                        extra={"note": "H2 stronger baselines; fresh streams +2000; M4 = strict 16-MVM variant",
                               "wall_s": time.time() - t0, "eval_device": device,
                               "full_registration": cfg["_variant"] == "full"},
                        side_tables=dict(oracle_raw=orc, cov_raw=cov, summary=wide_table(df)))  # fmt: skip
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
