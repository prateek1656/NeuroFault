# ruff: noqa: C408
"""Direction A follow-up: field-fit calibration (M4) for BatchNorm-folded deployment.

Registered in criteria/a_bnfold.yaml (frozen 2026-10-09, after Exp 2 / L16). W2 = BN folded into
the conv weights + a digital bias, ResNet-9, CIFAR-10 and CIFAR-100, 3 ckpt seeds each. Methods:
B8 (per-tile best of GDC/row/col/bi + mean-bias, chosen by held-out output error, same
as Exp 2), M4 = FieldFit('cos', order 4) with 8 fwd + 8 bwd probe MVMs per tile + mean-bias, and the
oracle = programmed (undrifted) network. Headline evaluation on 10k test images.

Fresh randomness: every random stream (programming noise, drift realisation, probe read noise,
probe vectors) is seeded with ckpt_seed + 1000 instead of ckpt_seed (Exp 2 used ckpt_seed):
ac.make_generator is wrapped to add the offset inside the TileCtx programming stream; the drift,
read and probe seeds are passed explicitly as ckpt_seed + 1000. Checkpoints themselves are loaded
by their real seed. CIFAR-100 checkpoints were never used in this vertical.

Conventions:
- m4_mvm_per_tile = counted FieldFit probe MVMs (fwd + bwd) per tile per update (16). The
  mean-bias correction needs one more read of the layer-mean input per tile for M4 and B8 alike
  (Exp 2 counted it in cost_mvm too); that read is NOT included in the column (16 + 1 = 17 total).
- CIFAR-100 activation caches (X_cache = im2col columns of the first 5000 CIFAR-100 test images,
  same recipe as P1.6) are computed here if missing, to results/spectral/activations_c100/.
  M4 does not need activations; B8's selection/bias do, exactly as for CIFAR-10.
- Gain CI: paired cluster bootstrap over ckpt seeds, per (dataset, t_s) pooled over scenarios.

  python experiments/spectral/a_bnfold.py --jobs 3 --allow-dirty [--smoke]
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import a_common as ac
import pandas as pd
import torch
from a_common import CONV_LAYERS, INCUMBENTS, TileCtx, TileRun, best_of, make_drift
from a_exp2_jobs import Exp2Bridge, _mk_get_tile
from f0_common import REPO_ROOT, finish, load_config, parse_args

from neurofault.spectral.baselines import GDC, BiScale, PerColumn, PerRow, bind_all
from neurofault.spectral.bnfold import SEED_OFFSET, add_gain_ci, tile_origins
from neurofault.spectral.harness import run_grid
from neurofault.spectral.methods_a import FieldFit
from neurofault.spectral.readout import TileReader
from neurofault.spectral.types import ReadNoise, make_generator

DATASETS = {"cifar10": 10, "cifar100": 100}


def ckpt_path(dataset: str, seed: int) -> Path:
    if dataset == "cifar10":
        return REPO_ROOT / "results" / "_live2" / f"resnet9_predeploy_seed{seed}" / "pre_deploy.pt"
    return REPO_ROOT / "results" / "_live5" / f"resnet9_predeploy_c100_seed{seed}" / "pre_deploy.pt"


def act_dir(dataset: str, seed: int) -> Path:
    sub = "activations" if dataset == "cifar10" else "activations_c100"
    return REPO_ROOT / "results" / "spectral" / sub / str(seed)


def _test_set(dataset: str, n: int):
    """First n test images, ToTensor + Normalize(0.5, 0.5). CIFAR-100 test comes from
    data/cifar100_hf/test.npz (HF uoft-cs/cifar100 test parquet, original order, first labels
    49,33,72,51,...; the toronto mirror was too slow); torchvision's loader is unavailable offline."""
    import numpy as np
    import torchvision

    if dataset == "cifar100":
        z = np.load(REPO_ROOT / "data" / "cifar100_hf" / "test.npz")
        x = torch.from_numpy(z["x"][:n]).permute(0, 3, 1, 2).float() / 255.0
        return (x - 0.5) / 0.5, torch.from_numpy(z["y"][:n]).long()
    T = torchvision.transforms
    ds = torchvision.datasets.CIFAR10(REPO_ROOT / "data", train=False, download=False,
             transform=T.Compose([T.ToTensor(), T.Normalize((0.5,) * 3, (0.5,) * 3)]))  # fmt: skip
    return torch.stack([ds[i][0] for i in range(n)]), torch.tensor([ds[i][1] for i in range(n)])


class BFBridge(Exp2Bridge):
    """W2 weight-space evaluation (BN folded into weights + bias) for either dataset."""

    def __init__(self, dataset: str, ckpt_seed: int, n_eval: int):
        from neurofault.models.cnn import SimpleCNN

        self.workload, self.dataset = "W2", dataset
        self.model = SimpleCNN(num_classes=DATASETS[dataset])
        blob = torch.load(ckpt_path(dataset, ckpt_seed), map_location="cpu", weights_only=False)
        self.model.load_state_dict(blob["model_state_dict"])
        self.model.eval()
        self.folded = {}
        self._fold()
        self.orig = {k: v.clone() for k, v in self.model.state_dict().items()}
        self.xe, self.ye = _test_set(dataset, n_eval)
        self.xc = None
        self._seed_key = ckpt_seed


_ACT: dict = {}


def _install(dataset: str, ckpt_seed: int, bridge: BFBridge):
    """Patch a_common: dataset-aware activation caches, folded tiles, +SEED_OFFSET programming."""
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


def ensure_c100_stats(seeds, n_images: int = 5000, positions: int = 16, cache_cols: int = 4096):
    """Compute (once) the im2col X_cache of CIFAR-100 test images per checkpoint. Only X_cache is
    consumed downstream (TileCtx); covariance spectra are not needed here."""
    from neurofault.models.cnn import SimpleCNN
    from neurofault.spectral.activations import RESNET9_LAYERS, capture_im2col_inputs

    x, y = _test_set("cifar100", n_images)
    for s in seeds:
        d = act_dir("cifar100", s)
        if all((d / f"{k.removesuffix('.weight')}.pt").exists() for k in RESNET9_LAYERS):
            continue
        m = SimpleCNN(num_classes=100)
        m.load_state_dict(
            torch.load(ckpt_path("cifar100", s), map_location="cpu", weights_only=False)[
                "model_state_dict"
            ]
        )
        loader = torch.utils.data.DataLoader(list(zip(x, y, strict=True)), batch_size=250)
        raw = capture_im2col_inputs(m, loader, RESNET9_LAYERS, n_images,
                                    make_generator(s, "activations_c100"),
                                    positions_per_image=positions, cache_cols=cache_cols)  # fmt: skip
        d.mkdir(parents=True, exist_ok=True)
        for k, v in raw.items():
            torch.save(dict(X_cache=v["X_cache"], n_images=v["n_images"], N=v["N"]),
                       d / f"{k.removesuffix('.weight')}.pt")  # fmt: skip
        print(f"c100 activations cached for seed {s}", flush=True)


class BFRun(TileRun):
    """Incumbents + M4 (8f+8b), sharing one read-noise stream."""

    def __init__(self, ctx: TileCtx, sigma: float, rseed: int):
        self.ctx, self.sigma = ctx, sigma
        self.cur = [ctx.stp]
        noise = ReadNoise("per_cell", sigma, adc_bits=None, dac_bits=None)
        self.reader = TileReader(
            lambda: self.cur[0], noise, make_generator(rseed, f"read:{ctx.key}")
        )
        sd = rseed + (sum(ord(ch) for ch in repr(ctx.key)) & 0xFFFF)
        self.methods = {
            "gdc": GDC("ls", p=8, seed=sd),
            "row": PerRow("rademacher", 8, "ls", seed=sd),
            "col": PerColumn("rademacher", 8, "backward", seed=sd),
            "bi": BiScale(8, 8, seed=sd),
            "m4": FieldFit("cos", order=4, p_fwd=8, p_bwd=8, seed=sd),
        }
        bind_all(list(self.methods.values()), lambda: self.cur[0].weights(), lambda: ctx.tile.w)
        self.cost = {k: m.calibrate(self.reader) for k, m in self.methods.items()}
        self.y0bar = self.reader.mvm(ctx.xbar[:, None])[:, 0]
        self.sketch = None
        # strict-budget M4 (8 fwd + 7 bwd = 15 probes, +1 bias read = 16 total) reads through its
        # OWN noise stream, so the main reader's draws (and thus b8 / m4) are bit-identical to the
        # first run that had no m4s.
        self.reader2 = TileReader(
            lambda: self.cur[0], noise, make_generator(rseed, f"read2:{ctx.key}")
        )
        self.m4s = FieldFit("cos", order=4, p_fwd=8, p_bwd=7, seed=sd)
        bind_all([self.m4s], lambda: self.cur[0].weights(), lambda: ctx.tile.w)
        self.m4s.calibrate(self.reader2)

    def at(self, t: float) -> dict:
        out = super().at(t)
        wt = out["_wt"]
        c = self.m4s.update(self.reader2, t)
        # same bias formula as TileRun.at (xread = this time's read of the layer-mean input)
        xread = self.reader2.mvm(self.ctx.xbar[:, None])[:, 0]
        b = self.y0bar - (xread + (c.W_eff - wt) @ self.ctx.xbar)
        out["m4s"] = dict(W=c.W_eff, b=b, meta=c.meta, cost=c.cost, err=self.ctx.err(c.W_eff, b))
        return out


def _tiles(bridge: BFBridge):
    return tile_origins(bridge.layer_mats(), CONV_LAYERS, 256, 16)


def oracle_job(p: dict) -> list[dict]:
    """Digital float and programmed (undrifted) accuracy of the folded network."""
    torch.set_num_threads(1)
    cfg, seed, ds = p["cfg"], p["seed"], p["dataset"]
    bridge = BFBridge(ds, seed, cfg["n_eval"])
    saved = _install(ds, seed, bridge)
    try:
        mats = bridge.layer_mats()
        pm = {}
        for layer, r0, c0, m, n in _tiles(bridge):
            ctx = TileCtx(seed, layer, r0, c0, m, n, 256)
            pm.setdefault(layer, torch.zeros_like(mats[layer], dtype=torch.float32))
            pm[layer][r0 : r0 + m, c0 : c0 + n] = ctx.wp.float()
        return [dict(dataset=ds, ckpt_seed=seed, acc_digital=bridge.acc(None),
                     acc_prog=bridge.acc(pm), n_eval=cfg["n_eval"], n_tiles=len(_tiles(bridge)))]  # fmt: skip
    finally:
        _restore(saved)


def net_job(p: dict) -> list[dict]:
    torch.set_num_threads(1)
    cfg, seed, ds, sc = p["cfg"], p["seed"], p["dataset"], p["scenario"]
    bridge = BFBridge(ds, seed, cfg["n_eval"])
    saved = _install(ds, seed, bridge)
    rseed = seed + SEED_OFFSET
    try:
        mats = bridge.layer_mats()
        mod = make_drift(sc["model"], sc.get("field") or "row_gradient", sc["rho"], 1.0)
        runs = []
        for layer, r0, c0, m, n in _tiles(bridge):
            ctx = TileCtx(seed, layer, r0, c0, m, n, 256)
            real = mod.sample(ctx.stp, make_generator(rseed, f"drift:{layer}:{r0}:{c0}"))
            run = BFRun(ctx.slim(), cfg["sigma"], rseed)
            run.real = real
            runs.append(((layer, r0, c0, m, n), run))
        # name -> (key chooser, uses mean-bias)
        specs = {"b8": (lambda r: best_of(r, INCUMBENTS), True), "m4": (lambda r: "m4", True),
                 "m4s": (lambda r: "m4s", True)}  # fmt: skip
        specs = {k: v for k, v in specs.items() if k in sc.get("evals", list(specs))}
        rows = []
        for tlabel, t in cfg["times"].items():
            W = {k: {} for k in specs}
            B = {k: {} for k in specs}
            mvm = {k: 0 for k in specs}
            for (layer, r0, c0, m, n), run in runs:
                res = run.at(t)
                for name, (choose, use_b) in specs.items():
                    key = choose(res)
                    r = res[key]
                    w = res["_wt"] if key == "none" else r["W"]
                    W[name].setdefault(layer, torch.zeros_like(mats[layer], dtype=torch.float32))
                    W[name][layer][r0 : r0 + m, c0 : c0 + n] = w.float()
                    if use_b and key != "none":
                        bd = B[name].setdefault(layer, torch.zeros(mats[layer].shape[0]))
                        bd[r0 : r0 + m] += r["b"].float()
                    if r.get("cost") is not None:
                        mvm[name] += int(r["cost"].mvm_fwd + r["cost"].mvm_bwd)
            n_t = len(runs)
            acc = {k: bridge.acc(W[k], B[k] if specs[k][1] else None) for k in specs}
            rows.append(dict(dataset=ds, ckpt_seed=seed, scenario=sc["name"], model=sc["model"],
                             field=sc.get("field") or "", rho=sc["rho"], time=tlabel, t_s=t,
                             acc_b8=acc.get("b8"), acc_m4=acc.get("m4"), acc_m4s=acc.get("m4s"),
                             m4_mvm_per_tile=mvm.get("m4", 0) / n_t,
                             m4s_mvm_per_tile=mvm.get("m4s", 0) / n_t,
                             b8_probe_mvm_per_tile=mvm.get("b8", 0) / n_t, n_tiles=n_t,
                             n_eval=cfg["n_eval"]))  # fmt: skip
            del W, B
        return rows
    finally:
        _restore(saved)


def build_metrics(net: pd.DataFrame, orc: pd.DataFrame) -> pd.DataFrame:
    df = net.merge(orc[["dataset", "ckpt_seed", "acc_prog", "acc_digital"]],
                   on=["dataset", "ckpt_seed"], how="left")  # fmt: skip
    df["gap_prog_minus_m4_pt"] = df.acc_prog - df.acc_m4
    df["gain_m4_minus_b8_pt"] = df.acc_m4 - df.acc_b8
    df = add_gain_ci(df, n_boot=4000, seed=1)
    return df


COLS = ["dataset", "ckpt_seed", "scenario", "model", "t_s", "acc_prog", "acc_m4", "acc_b8",
        "gap_prog_minus_m4_pt", "gain_m4_minus_b8_pt", "gain_ci_lower_pt", "m4_mvm_per_tile"]  # fmt: skip


def main(argv=None) -> int:
    a = parse_args(__doc__.splitlines()[0], "a_bnfold.yaml", "a_bnfold", argv)
    cfg = load_config(a.config, a.smoke)
    cfg.update(master_seed=a.seed, seed_offset=SEED_OFFSET)
    out = Path(a.out)
    if "cifar100" in cfg["datasets"]:
        ensure_c100_stats(cfg["ckpt_seeds"])
    ocfg = {k: v for k, v in cfg.items() if k != "scenarios"}
    opts = [dict(cfg=ocfg, seed=s, dataset=d) for d in cfg["datasets"] for s in cfg["ckpt_seeds"]]
    pts = [dict(cfg=cfg, seed=s, dataset=d, scenario=sc)
           for sc in cfg["scenarios"] for d in cfg["datasets"] for s in cfg["ckpt_seeds"]]  # fmt: skip
    if os.environ.get("A_BNFOLD_CACHE_ONLY"):
        import json

        def rd(p):
            return pd.DataFrame([r for ln in (p / "cells.jsonl").read_text().splitlines() if ln.strip()
                                 for r in json.loads(ln)["rows"]])  # fmt: skip

        orc, net = rd(out / "_cells_oracle"), rd(out / "_cells_net")
    else:
        orc = run_grid(oracle_job, opts, jobs=a.jobs, out_dir=out / "_cells_oracle")
        net = run_grid(net_job, pts, jobs=a.jobs, out_dir=out / "_cells_net")
    key = ["dataset", "ckpt_seed", "scenario", "t_s"]
    reuse = os.environ.get(
        "A_BNFOLD_REUSE"
    )  # earlier run's _cells_net: supplies b8/m4 of scenarios
    if reuse:  # whose job only evaluated the strict variant (read streams are bit-identical)
        import json

        old = pd.DataFrame([r for ln in (Path(reuse) / "cells.jsonl").read_text().splitlines()
                            if ln.strip() for r in json.loads(ln)["rows"]])  # fmt: skip
        o = old.set_index(key)[["acc_b8", "acc_m4"]]
        net = net.set_index(key)
        both = net.acc_m4.notna() & net.index.isin(o.index)
        chk = (net.acc_m4[both] - o.acc_m4.reindex(net.index[both])).abs()
        print(f"determinism check vs reused cells: {int(both.sum())} rows, max |d acc_m4| = "
              f"{chk.max():.6f}", flush=True)  # fmt: skip
        for c in ("acc_b8", "acc_m4"):
            net[c] = net[c].fillna(o[c].reindex(net.index))
        net = net.reset_index()
        net["m4_mvm_per_tile"] = net.m4_mvm_per_tile.where(net.m4_mvm_per_tile > 0, 16.0)
    assert net[["acc_b8", "acc_m4", "acc_m4s"]].notna().all().all(), "missing evals"
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    # A = M4 as registered-run (8f+8b probes: 16 counted MVMs; +1 bias read = 17 total)
    full = build_metrics(net, orc)
    # B = strict M4 (8f+7b probes + 1 bias read = 16 total per tile)
    netb = net.copy()
    netb["acc_m4"] = netb.acc_m4s
    netb["m4_mvm_per_tile"] = netb.m4s_mvm_per_tile + 1.0
    fullb = build_metrics(netb, orc)
    fullt = full.copy()  # A with the bias read counted: 17 total
    fullt["m4_mvm_per_tile"] = fullt.m4_mvm_per_tile + 1.0

    def summ(f):
        return f.groupby(["dataset", "t_s"]).agg(
            acc_prog=("acc_prog", "mean"), acc_m4=("acc_m4", "mean"), acc_b8=("acc_b8", "mean"),
            gain=("gain_m4_minus_b8_pt", "mean"), ci_lo=("gain_ci_lower_pt", "first"),
        ).reset_index()  # fmt: skip

    print("== A: M4 8f+8b (16 probe MVMs, 17 with bias) ==")
    print(summ(full).round(3).to_string(index=False))
    print("== B: strict M4 8f+7b+bias (16 total) ==")
    print(summ(fullb).round(3).to_string(index=False))
    run_dir, _ = finish("A-BNfold(A: 16 probes)", "a_bnfold.yaml", cfg, a, full[COLS],
                        extra={"note": "BN-folded M4 follow-up; variant A = 8f+8b probes"},
                        side_tables=dict(net_raw=full, oracle_raw=orc, summary=summ(full),
                                         net_raw_strict=fullb, summary_strict=summ(fullb),
                                         metrics_A_total17=fullt[COLS], metrics_B_strict16=fullb[COLS]))  # fmt: skip
    import f0_common as fc

    crit = fc.load_criteria(fc.CRITERIA_DIR / "a_bnfold.yaml")
    for tag, f in (("A_total17", fullt), ("B_strict16", fullb)):
        v = fc.evaluate(crit, f[COLS])
        fc.write_verdict(v, run_dir / f"verdict_{tag}.json", extra={"variant": tag})
        print(f"\n=== verdict {tag}: {v['overall']} ===")
        for c in v["criteria"]:
            print(f"  {c['id']:<14s} {c['kind']:<8s} {c['status']:<13s} "
                  f"{fc._leaf_values(c.get('detail', {}))}")  # fmt: skip
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
