# ruff: noqa: C408
"""Phase H / H6: M4 as a refresh scheduler (criteria/a_refresh.yaml, frozen 2026-10-10).

Platform = a_deploy (BN-folded ResNet-9 W2, CIFAR-10/100, ckpt seeds 42/101/102, 8 scenarios, 2% per-cell probe
read noise, 10k-image eval, train-image statistics) with every random stream offset +4000 (a_deploy's engine is
reused with its module offsets rebound). Steady-state per-tile refresh model: tile i is reprogrammed at period
T_i in the age grid; its worst-case state is its (single) drift realization at age T_i; network worst-case
accuracy = every tile at its own T_i at once. Served correction between refreshes = B8 for all families.
Families: TIMER(T); M4-ALLOC(lam) / GDC-ALLOC(lam): per-tile period T_i = argmin_k wgt_i h_i(k)^2 + lam writes_i(k),
with h_i the tile's statistic curve extrapolated log-linearly from a 1-day commissioning window; ORACLE-ALLOC (report)
uses the true served-B8 error curve. See the criteria note for the full definition.

  python experiments/spectral/a_refresh.py --jobs 3 --allow-dirty [--smoke]
"""

from __future__ import annotations

import itertools
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import a_deploy as ad

SEED_OFFSET = 4000
ad.SEED_OFFSET = SEED_OFFSET
ad.POOL_SEED_BASE = 7000 + SEED_OFFSET

import numpy as np
import pandas as pd
import torch
from a_bnfold import _tiles
from a_common import INCUMBENTS, TileCtx, best_of, make_drift
from f0_common import finish, load_config, parse_args

from neurofault.spectral.harness import run_grid
from neurofault.spectral.methods_a import (
    cosine_basis,
    fit_basis_field,
    project_cosine_coeffs,
    rank_factors,
)
from neurofault.spectral.types import make_generator

YEAR = 31536000.0


def m4_health(run: ad.DRun, wt: torch.Tensor, sigma: float) -> tuple[float, float]:
    """(h_m4, sigma_eps_hat): predicted relative output error of a served gain-type correction, from M4's
    16 reads only (W0, probes, read-noise spec, train-image activations; never W(t) or the true error)."""
    m4, rd, ctx = run.m4, run.reader2, run.ctx
    df = rd.mvm(m4.xf) - m4.y0
    db = rd.mvm_t(m4.zb) - m4.z0
    w0 = m4.ref()
    m, n = w0.shape
    if getattr(run, "_basis", None) is None:
        run._basis = cosine_basis(m, n, 4)
    f = fit_basis_field(w0, run._basis, m4.xf, df, m4.zb, db, m4.ridge)
    wf = w0 * (1.0 + f)
    rf = df - (w0 * f) @ m4.xf
    rb = db - (w0 * f).T @ m4.zb
    pf, pb = m4.xf.shape[1], m4.zb.shape[1]
    s = sigma * float(wf.abs().max())  # per-cell read-noise std (spec x full scale, estimated)
    noise_e = (
        2.0 * s * s * m * n * (pf + pb)
    )  # E||read noise||^2 in both residuals (two reads per diff)
    sig_e2 = max(
        0.0,
        (float(rf.pow(2).sum() + rb.pow(2).sum()) - noise_e) / ((pf + pb) * float(wf.pow(2).sum())),
    )
    g = 1.0 / (1.0 + f).clamp_min(0.1)
    a_, b_ = rank_factors(project_cosine_coeffs(g, 4), (m, n), 1)
    g1 = a_ @ b_.T
    x = ctx.x_eval
    struct = (w0 * ((1.0 + f) * g1 - 1.0)) @ x
    iid = sig_e2 * float(((wf * g1).pow(2) @ x.pow(2)).sum())
    ref = float((w0 @ x).pow(2).sum())
    return float(np.sqrt((float(struct.pow(2).sum()) + iid) / ref)), float(np.sqrt(sig_e2))


def net_job(p: dict) -> list[dict]:
    torch.set_num_threads(1)
    cfg, seed, ds, sc = p["cfg"], p["seed"], p["dataset"], p["scenario"]
    bridge = ad.DBridge(ds, seed, cfg["n_eval"])
    saved = ad._install(cfg, ds, bridge)
    rseed = seed + SEED_OFFSET
    ages = list(cfg["ages"].items())
    na = len(ages)
    try:
        mats = bridge.layer_mats()
        mod = make_drift(sc["model"], sc.get("field") or "row_gradient", sc["rho"], 1.0)
        tiles, Wb8, Bb8 = [], [], []
        stat = {
            k: np.zeros((0, na)) for k in ("m4", "gdc", "oracle")
        }  # + "static" (added after the tile loop)
        sig_eps, ref_energy = [], []
        for layer, r0, c0, m, n in _tiles(bridge):
            ctx = TileCtx(seed, layer, r0, c0, m, n, 256)
            run = ad.DRun(ctx.slim(), cfg["sigma"], rseed, None)
            run.real = mod.sample(ctx.stp, make_generator(rseed, f"drift:{layer}:{r0}:{c0}"))
            ws, bs, row = [], [], {k: [] for k in stat}
            se = []
            for _lab, t in ages:
                run.cur[0] = run.real.at(t)
                wt = run.cur[0].weights()
                xread = run.reader.mvm(ctx.xbar[:, None])[:, 0]
                res = {}
                for k, meth in run.methods.items():
                    c = meth.update(run.reader, t)
                    b = run._bias(c.W_eff, xread, wt)
                    res[k] = dict(W=c.W_eff, b=b, err=ctx.err(c.W_eff, b), meta=c.meta)
                best = res[best_of(res, INCUMBENTS)]
                ws.append(best["W"].to(torch.float16))
                bs.append(best["b"].float())
                row["oracle"].append(best["err"])
                alpha = float(torch.as_tensor(res["gdc"]["meta"]["alpha"]).mean())
                row["gdc"].append(abs(1.0 - 1.0 / alpha) if alpha != 0 else float("inf"))
                h, s_ = m4_health(run, wt, cfg["sigma"])
                row["m4"].append(h)
                se.append(s_)
            for k, prev in stat.items():
                stat[k] = np.vstack([prev, np.asarray(row[k])[None]])
            tiles.append((layer, r0, c0, m, n))
            ref_energy.append(float((ctx.tile.w @ ctx.x_eval).pow(2).sum()))
            Wb8.append(ws)
            Bb8.append(bs)
            sig_eps.append(se)
        cells = np.array([m * n for (_l, _r, _c, m, n) in tiles], dtype=float)
        age_s = np.array([t for _l, t in ages])

        def assemble(k_idx: np.ndarray) -> float:
            W, B = {}, {}
            for i, (layer, r0, c0, m, n) in enumerate(tiles):
                ad._put(W, mats, layer, r0, c0, m, n, Wb8[i][k_idx[i]].float())
                bd = B.setdefault(layer, torch.zeros(mats[layer].shape[0]))
                bd[r0 : r0 + m] += Bb8[i][k_idx[i]]
            return bridge.acc(W, B)

        def writes(k_idx: np.ndarray) -> float:
            return float((cells * (YEAR / age_s[k_idx])).sum() / cells.sum())

        base = dict(dataset=ds, ckpt_seed=seed, scenario=sc["name"], model=sc["model"],
                    field=sc.get("field") or "", rho=sc["rho"], n_tiles=len(tiles), n_eval=cfg["n_eval"])  # fmt: skip
        rows = []
        for j, (lab, _t) in enumerate(ages):
            k = np.full(len(tiles), j)
            rows.append(dict(base, family="timer", param=lab, idx=j, writes=writes(k), acc=assemble(k),
                             checks_per_cycle=0.0))  # fmt: skip
        # per-tile output-energy weight within its layer (label-free); each layer counts equally
        lay = [t[0] for t in tiles]
        ref_e = np.array(ref_energy)
        wgt = np.array([ref_e[i] / sum(ref_e[j] for j in range(len(tiles)) if lay[j] == lay[i])
                        for i in range(len(tiles))])  # fmt: skip
        logs = np.log(age_s)
        comm = logs <= np.log(cfg["commission_s"]) + 1e-9

        def curve(s: np.ndarray, extrapolate: bool) -> np.ndarray:
            """Per-tile predicted statistic at every grid age: log-linear fit on the commissioning ages."""
            if not extrapolate:
                return s
            x = logs[comm]
            xm = x.mean()
            b = ((s[:, comm] - s[:, comm].mean(1, keepdims=True)) * (x - xm)).sum(1) / (
                (x - xm) ** 2
            ).sum()
            a = s[:, comm].mean(1) - b * xm
            return np.clip(a[:, None] + b[:, None] * logs[None, :], 0.0, None)

        def allocate(h: np.ndarray, lam: float) -> np.ndarray:
            """T_i = argmin_k  wgt_i h_i(k)^2 + lam * cells_i/sum(cells) * (YEAR / age_k)."""
            cost = (
                wgt[:, None] * h**2 + lam * (cells / cells.sum())[:, None] * (YEAR / age_s)[None, :]
            )
            return cost.argmin(1)

        lams = np.logspace(cfg["lam_log10"][0], cfg["lam_log10"][1], cfg["n_lam"])
        # sensorless STATIC allocation: one nominal age curve for every tile (PCM, mean nu 0.06, t0 20 s)
        stat["static"] = np.broadcast_to(1.0 - (age_s / 20.0) ** -0.06, stat["m4"].shape).copy()
        for fam, extrap in (("m4", True), ("gdc", True), ("static", False), ("oracle", False)):
            h = curve(stat[fam], extrap)
            for li, lam in enumerate(lams):
                k = allocate(h, float(lam))
                rows.append(dict(base, family=fam, param=f"lam{lam:.2e}", idx=li, tau=float(lam), writes=writes(k),
                                 acc=assemble(k),
                                 checks_per_cycle=float((cells * (k + 1) * (YEAR / age_s[k])).sum() / cells.sum()),
                                 mean_age_d=float((cells * age_s[k]).sum() / cells.sum() / 86400.0)))  # fmt: skip
        for i, (layer, r0, c0, _m, _n) in enumerate(tiles):
            rows.append(dict(base, family="_tile", param=f"{layer}:{r0}:{c0}", idx=i,
                             **{f"h_m4_{lab}": stat["m4"][i, j] for j, (lab, _t) in enumerate(ages)},
                             **{f"h_gdc_{lab}": stat["gdc"][i, j] for j, (lab, _t) in enumerate(ages)},
                             **{f"err_{lab}": stat["oracle"][i, j] for j, (lab, _t) in enumerate(ages)},
                             **{f"sigeps_{lab}": sig_eps[i][j] for j, (lab, _t) in enumerate(ages)}))  # fmt: skip
        return rows
    finally:
        ad._restore(saved)


def writes_to_reach(fr: pd.DataFrame, target: float, cap: float = 1.0e6) -> float:
    """Min writes on the (writes, acc) Pareto frontier reaching `target`; linear in log(writes)."""
    pts = fr.sort_values("writes")[["writes", "acc"]].to_numpy()
    front, best = [], -np.inf
    for w, a in pts:
        if a > best:
            front.append((w, a))
            best = a
    if not front or front[-1][1] < target:
        return cap
    if front[0][1] >= target:
        return float(front[0][0])
    for (w0, a0), (w1, a1) in itertools.pairwise(front):
        if a0 < target <= a1:
            u = (target - a0) / (a1 - a0)
            return float(np.exp(np.log(w0) + u * (np.log(w1) - np.log(w0))))
    return cap


def build_metrics(net: pd.DataFrame, targets: dict) -> pd.DataFrame:
    pol = net[net.family != "_tile"]
    out = []
    for (ds, seed, sc), g in pol.groupby(["dataset", "ckpt_seed", "scenario"]):
        tm = g[g.family == "timer"].set_index("param")
        for tname, tlab in targets.items():
            tgt, w_t = float(tm.loc[tlab, "acc"]), float(tm.loc[tlab, "writes"])
            r = dict(dataset=ds, ckpt_seed=seed, scenario=sc, model=g.model.iloc[0], target=tname,
                     target_acc=tgt, writes_timer=w_t)  # fmt: skip
            for fam in ("m4", "gdc", "static", "oracle"):
                r[f"writes_{fam}"] = writes_to_reach(g[g.family == fam], tgt)
            r["ratio_m4_timer"] = r["writes_m4"] / w_t
            r["ratio_m4_gdc"] = r["writes_m4"] / r["writes_gdc"]
            r["ratio_m4_static"] = r["writes_m4"] / r["writes_static"]
            r["ratio_static_timer"] = r["writes_static"] / w_t
            r["ratio_oracle_timer"] = r["writes_oracle"] / w_t
            r["monitor_mvm_per_tile_yr_m4"] = 16.0 * float(
                g[g.family == "m4"].checks_per_cycle.median()
            )
            r["monitor_mvm_per_tile_yr_gdc"] = 9.0 * float(
                g[g.family == "gdc"].checks_per_cycle.median()
            )
            out.append(r)
    return pd.DataFrame(out)


def main(argv=None) -> int:
    a = parse_args(__doc__.splitlines()[0], "a_refresh.yaml", "a_refresh", argv)
    cfg = load_config(a.config, a.smoke)
    out = Path(a.out)
    cfg.update(master_seed=a.seed, cache_dir=str(out / "_cache"))
    ocfg = {k: v for k, v in cfg.items() if k != "scenarios"}
    opts = [dict(cfg=ocfg, seed=s, dataset=d) for d in cfg["datasets"] for s in cfg["ckpt_seeds"]]
    run_grid(ad.act_job, opts, jobs=a.jobs, out_dir=out / "_cells_act")
    orc = run_grid(ad.oracle_job, opts, jobs=a.jobs, out_dir=out / "_cells_oracle")
    pts = [dict(cfg=cfg, seed=s, dataset=d, scenario=sc)
           for sc in cfg["scenarios"] for d in cfg["datasets"] for s in cfg["ckpt_seeds"]]  # fmt: skip
    net = run_grid(net_job, pts, jobs=a.jobs, out_dir=out / "_cells_net")
    met = build_metrics(net, cfg["targets"])
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    print(met.groupby(["dataset", "target"])[["writes_timer", "writes_m4", "writes_gdc", "writes_oracle",
                                              "writes_static", "ratio_m4_timer", "ratio_m4_gdc",
                                              "ratio_m4_static", "ratio_oracle_timer"]]
          .median().round(3).to_string())  # fmt: skip
    pol = net[net.family != "_tile"]
    fr = pol.groupby(["dataset", "family", "idx"])[["writes", "acc"]].mean().reset_index()
    run_dir, _ = finish("A-refresh(H6)", "a_refresh.yaml", cfg, a, met,
                        extra={"note": "H6 refresh scheduling; streams +4000; steady-state per-tile model"},
                        side_tables=dict(policies_raw=pol, tiles_raw=net[net.family == "_tile"], frontiers=fr,
                                         oracle=orc))  # fmt: skip
    print(f"run dir: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
