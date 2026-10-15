# ruff: noqa: C408
"""Phase H / H5: deployable cost of M4 (criteria/a_deploy.yaml, frozen 2026-10-10, commit 37af453).

Platform = a_bnfold (BN-folded ResNet-9 W2, CIFAR-10/100, ckpt seeds 42/101/102, 8 scenarios, t in
{1h,1d,30d,1yr}, 2% per-cell probe read noise, 10k-image eval) with ALL random streams offset +3000.
Per (dataset, seed, scenario, t), methods:
  B8       per-tile best of GDC/row/col/bi + mean-bias, chosen by held-out error (optimistic incumbent)
  M4-r1/r2/full   strict M4 field fit (cos order 4, 8 fwd + 7 bwd probes + 1 mean-bias read = 16 MVMs/tile)
                  applied in its DEPLOYABLE form W1 o G_r (methods_a.deployable_correction): G = 1/(1+F_hat)
                  projected on the 4x4 cosine span and truncated to rank r by SVD. Each rank-1 term is one
                  analog pass with input/output scaling in the digital periphery; r terms = r analog passes =
                  (r - 1) EXTRA passes (rank 1 costs nothing beyond the base MVM). "full" = no truncation (<= 4).
  REPROG   every tile is reprogrammed to its digital target at time t: fresh one-shot programming-noise draw,
           drift clock reset (read right after programming); cost = 1 write per device per reprogram.
  M4-ideal (diagnostic, not registered) W(t) - W0 o F_hat with the DIGITAL reference W0 (the earlier a_bnfold
           'm4s' semantics): costs a full digital MVM per inference, so it is an upper reference only.
Train-image statistics (DELIBERATE deviation from a_bnfold, per the registration): X_cache / xbar / covariances /
B8's held-out selection error come from AUGMENTED TRAIN images (crop pad 4 + flip + cutout), never test images.
Ablation (report only; CIFAR-10, t in {1d, 1yr}, all scenarios/seeds): M4 order {1..4} x {fwd-only: 15 fwd probes,
fwd+bwd: 8 fwd + 7 bwd}, each in the ideal form and the rank-1 deployable form.

  python experiments/spectral/a_deploy.py --jobs 3 --allow-dirty [--smoke]
"""

from __future__ import annotations

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
from a_bnfold import BFBridge, _tiles
from a_common import CONV_LAYERS, INCUMBENTS, PROG, TileCtx, best_of, make_drift
from a_exp2_jobs import POSITIONS, _mk_get_tile
from f0_common import REPO_ROOT, finish, load_config, parse_args

from neurofault.spectral.activations import RESNET9_LAYERS, capture_im2col_inputs
from neurofault.spectral.baselines import GDC, BiScale, PerColumn, PerRow, bind_all
from neurofault.spectral.exp2 import cluster_bootstrap_ci
from neurofault.spectral.harness import run_grid
from neurofault.spectral.methods_a import (
    FieldFit,
    deployable_correction,
    project_cosine_coeffs,
    rank_factors,
)
from neurofault.spectral.noise import program
from neurofault.spectral.readout import TileReader
from neurofault.spectral.types import ReadNoise, make_generator

SEED_OFFSET = 3000
POOL_SEED_BASE = 7000 + SEED_OFFSET
_DEVICE = os.environ.get("A_DEPLOY_DEVICE", "mps")
_ACT: dict = {}


# ------------------------------------------------------------------------------- data / platform


def train_pool(dataset: str, seed: int, n: int) -> torch.Tensor:
    """n AUGMENTED train images (the checkpoints' training augmentation), fixed per (dataset, seed)."""
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
        get, total = (lambda i: ds[i][0]), len(ds)
    else:
        from PIL import Image

        xs = np.load(REPO_ROOT / "data" / "cifar100_hf" / "train.npz")["x"]
        get, total = (lambda i: aug(Image.fromarray(xs[i]))), len(xs)
    idx = torch.randperm(total, generator=g)[:n].tolist()
    torch.manual_seed(POOL_SEED_BASE + seed)  # augmentation draws
    return torch.stack([get(i) for i in idx])


def act_cache(cfg: dict, dataset: str, seed: int) -> Path:
    return Path(cfg["cache_dir"]) / "acts" / f"{dataset}_{seed}"


def act_job(p: dict) -> list[dict]:
    """Train-image im2col X_cache of every conv layer (digital folded network == original function)."""
    torch.set_num_threads(1)
    cfg, seed, ds = p["cfg"], p["seed"], p["dataset"]
    d = act_cache(cfg, ds, seed)
    layers = {k: v for k, v in RESNET9_LAYERS.items() if k in CONV_LAYERS}
    if all((d / f"{k.removesuffix('.weight')}.pt").exists() for k in layers):
        return [dict(dataset=ds, ckpt_seed=seed, act_images=cfg["act_images"], cached=True)]
    t0 = time.time()
    bridge = BFBridge(ds, seed, 10)
    n = cfg["act_images"]
    imgs = train_pool(ds, seed, n)
    loader = [(imgs[i : i + 250],) for i in range(0, n, 250)]
    raw = capture_im2col_inputs(bridge.model.eval(), loader, layers, n,
                                make_generator(seed + SEED_OFFSET, "activations_train"),
                                positions_per_image=16, cache_cols=4096)  # fmt: skip
    d.mkdir(parents=True, exist_ok=True)
    for k, v in raw.items():
        torch.save(dict(X_cache=v["X_cache"], n_images=v["n_images"], N=v["N"]),
                   d / f"{k.removesuffix('.weight')}.pt")  # fmt: skip
    return [dict(dataset=ds, ckpt_seed=seed, act_images=n, cached=False, wall_s=time.time() - t0)]


class DBridge(BFBridge):
    """BFBridge with the 10k-image evaluation on `device` (same forward for every method)."""

    def __init__(self, dataset: str, ckpt_seed: int, n_eval: int):
        super().__init__(dataset, ckpt_seed, n_eval)
        self.dev = _DEVICE if (_DEVICE != "mps" or torch.backends.mps.is_available()) else "cpu"
        self.model.to(self.dev)
        self.xe, self.ye = self.xe.to(self.dev), self.ye.to(self.dev)

    @torch.no_grad()
    def acc(self, mats, bias=None, ada_n: int = 0) -> float:
        from a_common import BN_OF

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


def _install(cfg: dict, dataset: str, bridge: BFBridge):
    saved = (ac.get_tile, ac.act_stats, ac.make_generator)
    ac.get_tile = _mk_get_tile(bridge)

    def act_stats(seed, layer):
        k = (dataset, seed, layer)
        if k not in _ACT:
            p = act_cache(cfg, dataset, seed) / f"{layer.removesuffix('.weight')}.pt"
            _ACT[k] = torch.load(p, weights_only=False)
        return _ACT[k]

    ac.act_stats = act_stats
    ac.make_generator = lambda seed, stream, *a, **k: make_generator(
        seed + SEED_OFFSET, stream, *a, **k
    )
    return saved


def _restore(saved):
    ac.get_tile, ac.act_stats, ac.make_generator = saved


# ------------------------------------------------------------------------------- per-tile engine


class DRun:
    """Incumbents (reader), strict M4 (reader2, same streams as a_bnfold's m4s) and optional ablation
    FieldFits (each its own reader). All estimators read through counted TileReaders."""

    def __init__(self, ctx: TileCtx, sigma: float, rseed: int, abl: dict | None):
        self.ctx = ctx
        self.cur = [ctx.stp]
        noise = ReadNoise("per_cell", sigma, adc_bits=None, dac_bits=None)
        mk = lambda tag: TileReader(
            lambda: self.cur[0], noise, make_generator(rseed, f"{tag}:{ctx.key}")
        )
        self.reader, self.reader2 = mk("read"), mk("read2")
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
        self.abl = {}
        for o in (abl or {}).get("orders", []):
            for tag, (pf, pb) in (("fb", (8, 7)), ("f", (abl["p_total"], 0))):
                if o == 4 and tag == "fb":
                    continue  # identical to the main strict M4
                rd = mk(f"abl{o}{tag}")
                ff = FieldFit("cos", order=o, p_fwd=pf, p_bwd=pb, seed=sd)
                bind_all([ff], w_now, w0)
                ff.calibrate(rd)
                self.abl[(o, tag)] = (ff, rd)

    def _bias(self, w_eff, xread, wt):
        return self.y0bar - (xread + (w_eff - wt) @ self.ctx.xbar)

    def at(self, t: float, do_abl: bool) -> dict:
        ctx = self.ctx
        self.cur[0] = self.real.at(t)
        wt = self.cur[0].weights()
        xread = self.reader.mvm(ctx.xbar[:, None])[:, 0]
        out = {"_wt": wt}
        for k, m in self.methods.items():
            c = m.update(self.reader, t)
            b = self._bias(c.W_eff, xread, wt)
            out[k] = dict(W=c.W_eff, b=b, cost=c.cost, err=ctx.err(c.W_eff, b))
        c = self.m4.update(self.reader2, t)
        x2 = self.reader2.mvm(ctx.xbar[:, None])[:, 0]
        f = c.meta["F"]
        out["m4ideal"] = dict(W=c.W_eff, b=self._bias(c.W_eff, x2, wt), cost=c.cost, passes=None)
        for name, rk in (("m4r1", 1), ("m4r2", 2), ("m4full", None)):
            w, terms = deployable_correction(wt, f, 4, rk)
            out[name] = dict(W=w, b=self._bias(w, x2, wt), cost=c.cost, passes=terms - 1)
        # report-only: literal first-order form W1 - sum_k diag(u_k) W1 diag(v_k) (k <= r, r extra passes)
        cf = project_cosine_coeffs(f, 4)
        for name, rk in (("m4r1_F", 1), ("m4full_F", 16)):
            a_, b_ = rank_factors(cf, tuple(f.shape), rk)
            w = wt - wt * (a_ @ b_.T)
            out[name] = dict(W=w, b=self._bias(w, x2, wt), cost=c.cost, passes=a_.shape[1])
        if do_abl:
            for (o, tag), (ff, rd) in self.abl.items():
                ca = ff.update(rd, t)
                xr = rd.mvm(ctx.xbar[:, None])[:, 0]
                out[f"abl{o}{tag}_ideal"] = dict(
                    W=ca.W_eff, b=self._bias(ca.W_eff, xr, wt), cost=ca.cost
                )
                w, _ = deployable_correction(wt, ca.meta["F"], o, 1)
                out[f"abl{o}{tag}_r1"] = dict(W=w, b=self._bias(w, xr, wt), cost=ca.cost)
            out["abl4fb_ideal"], out["abl4fb_r1"] = out["m4ideal"], out["m4r1"]
        return out


def oracle_job(p: dict) -> list[dict]:
    torch.set_num_threads(1)
    cfg, seed, ds = p["cfg"], p["seed"], p["dataset"]
    bridge = DBridge(ds, seed, cfg["n_eval"])
    saved = _install(cfg, ds, bridge)
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


def _put(W, mats, layer, r0, c0, m, n, w):
    W.setdefault(layer, torch.zeros_like(mats[layer], dtype=torch.float32))
    W[layer][r0 : r0 + m, c0 : c0 + n] = w.float()


def net_job(p: dict) -> list[dict]:
    torch.set_num_threads(1)
    cfg, seed, ds, sc = p["cfg"], p["seed"], p["dataset"], p["scenario"]
    ab = cfg.get("ablation") if ds == cfg["ablation"]["dataset"] else None
    bridge = DBridge(ds, seed, cfg["n_eval"])
    saved = _install(cfg, ds, bridge)
    rseed = seed + SEED_OFFSET
    try:
        mats = bridge.layer_mats()
        mod = make_drift(sc["model"], sc.get("field") or "row_gradient", sc["rho"], 1.0)
        runs = []
        for layer, r0, c0, m, n in _tiles(bridge):
            ctx = TileCtx(seed, layer, r0, c0, m, n, 256)
            real = mod.sample(ctx.stp, make_generator(rseed, f"drift:{layer}:{r0}:{c0}"))
            run = DRun(ctx.slim(), cfg["sigma"], rseed, ab)
            run.real = real
            runs.append(((layer, r0, c0, m, n), ctx, run))
        rows = []
        for tlabel, t in cfg["times"].items():
            do_abl = ab is not None and tlabel in ab["times"]
            names = ["b8", "m4r1", "m4r2", "m4full", "m4ideal", "m4r1_F", "m4full_F", "reprog"]
            if do_abl:
                names += [
                    f"abl{o}{g}_{f}"
                    for o in ab["orders"]
                    for g in ("fb", "f")
                    for f in ("ideal", "r1")
                ]
            W = {k: {} for k in names}
            B = {k: {} for k in names}
            mvm = dict.fromkeys(("b8", "m4"), 0)
            passes = {"m4r1": [], "m4r2": [], "m4full": [], "m4r1_F": [], "m4full_F": []}
            wpos = dict.fromkeys(passes, 0.0)
            pos_tot = 0.0
            for (layer, r0, c0, m, n), ctx, run in runs:
                res = run.at(t, do_abl)
                for name in names:
                    if name == "reprog":
                        g = make_generator(rseed, f"reprog:{run.ctx.key}:{t}")
                        _put(
                            W["reprog"],
                            mats,
                            layer,
                            r0,
                            c0,
                            m,
                            n,
                            program(run.ctx.st0, PROG, g).weights(),
                        )
                        continue
                    r = res[best_of(res, INCUMBENTS)] if name == "b8" else res[name]
                    _put(W[name], mats, layer, r0, c0, m, n, r["W"])
                    bd = B[name].setdefault(layer, torch.zeros(mats[layer].shape[0]))
                    bd[r0 : r0 + m] += r["b"].float()
                    if name == "b8":
                        mvm["b8"] += int(r["cost"].mvm_fwd + r["cost"].mvm_bwd)
                    if name == "m4ideal":
                        mvm["m4"] += int(r["cost"].mvm_fwd + r["cost"].mvm_bwd)
                pos_tot += POSITIONS[layer]
                for k, plist in passes.items():
                    plist.append(res[k]["passes"])
                    wpos[k] += POSITIONS[layer] * (1 + res[k]["passes"])
            n_t = len(runs)
            acc = {k: bridge.acc(W[k], B[k] if k != "reprog" else None) for k in names}
            row = dict(dataset=ds, ckpt_seed=seed, scenario=sc["name"], model=sc["model"],
                       field=sc.get("field") or "", rho=sc["rho"], time=tlabel, t_s=t, n_tiles=n_t,
                       n_eval=cfg["n_eval"],
                       # calibration cost (counted probe MVMs per tile per update, +1 mean-bias read for both)
                       mvm_cal_b8=mvm["b8"] / n_t + 1, mvm_cal_m4=mvm["m4"] / n_t + 1,
                       extra_passes_b8=0, extra_passes_reprog=0, extra_passes_m4r1=max(passes["m4r1"]),
                       extra_passes_m4r2=max(passes["m4r2"]), extra_passes_m4full=max(passes["m4full"]),
                       extra_passes_m4r1_F=max(passes["m4r1_F"]), extra_passes_m4full_F=max(passes["m4full_F"]),
                       extra_passes_m4full_mean=float(np.mean(passes["m4full"])),
                       extra_passes_m4r2_mean=float(np.mean(passes["m4r2"])),
                       inference_mvm_factor_m4r2=wpos["m4r2"] / pos_tot,
                       inference_mvm_factor_m4full=wpos["m4full"] / pos_tot,
                       writes_per_cell_reprog=1)  # fmt: skip
            for k, v in acc.items():
                row[f"acc_{k}" if not k.startswith("abl") else f"abl_{k[3:]}"] = v
            rows.append(row)
            del W, B
        return rows
    finally:
        _restore(saved)


# ------------------------------------------------------------------------------- metrics


def build_metrics(net: pd.DataFrame, orc: pd.DataFrame) -> pd.DataFrame:
    df = net.merge(orc[["dataset", "ckpt_seed", "acc_prog", "acc_digital"]],
                   on=["dataset", "ckpt_seed"], how="left")  # fmt: skip
    df["gain_m4r1_minus_b8_pt"] = df.acc_m4r1 - df.acc_b8
    df["gain_m4r1_ci_lower_pt"] = np.nan
    for idx in df.groupby(["dataset", "t_s"]).groups.values():
        g = df.loc[idx]
        _, lo, _ = cluster_bootstrap_ci(
            g.gain_m4r1_minus_b8_pt.to_numpy(), g.ckpt_seed.to_numpy(), 4000, 0.05, 1
        )
        df.loc[idx, "gain_m4r1_ci_lower_pt"] = lo
    return df


def main(argv=None) -> int:
    a = parse_args(__doc__.splitlines()[0], "a_deploy.yaml", "a_deploy", argv)
    cfg = load_config(a.config, a.smoke)
    out = Path(a.out)
    cfg.update(master_seed=a.seed, cache_dir=str(out / "_cache"))
    ocfg = {k: v for k, v in cfg.items() if k != "scenarios"}
    opts = [dict(cfg=ocfg, seed=s, dataset=d) for d in cfg["datasets"] for s in cfg["ckpt_seeds"]]
    run_grid(act_job, opts, jobs=a.jobs, out_dir=out / "_cells_act")
    orc = run_grid(oracle_job, opts, jobs=a.jobs, out_dir=out / "_cells_oracle")
    pts = [dict(cfg=cfg, seed=s, dataset=d, scenario=sc)
           for sc in cfg["scenarios"] for d in cfg["datasets"] for s in cfg["ckpt_seeds"]]  # fmt: skip
    net = run_grid(net_job, pts, jobs=a.jobs, out_dir=out / "_cells_net")
    full = build_metrics(net, orc)
    abl_cols = [c for c in full.columns if c.startswith("abl_")]
    metrics_cols = [c for c in full.columns if c not in abl_cols]
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    meth = [
        "acc_prog",
        "acc_b8",
        "acc_m4r1",
        "acc_m4r2",
        "acc_m4full",
        "acc_reprog",
        "acc_m4ideal",
        "acc_m4r1_F",
        "acc_m4full_F",
    ]
    summ = (
        full.groupby(["dataset", "t_s"])[meth + ["gain_m4r1_minus_b8_pt", "gain_m4r1_ci_lower_pt"]]
        .mean()
        .reset_index()
    )
    print(summ.round(3).to_string(index=False))
    side = dict(summary=summ, net_raw=full)
    if abl_cols:
        ab = full[full[abl_cols].notna().any(axis=1)]
        side["ablation"] = (
            ab.groupby(["dataset", "t_s"])[abl_cols + ["acc_prog", "acc_b8"]].mean().reset_index()
        )
        print(side["ablation"].round(3).T.to_string())
    run_dir, _ = finish("A-deploy(H5)", "a_deploy.yaml", cfg, a, full[metrics_cols],
                        extra={"note": "H5 deployable cost of M4; streams +3000; train-image statistics"},
                        side_tables=side)  # fmt: skip
    print(f"run dir: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
