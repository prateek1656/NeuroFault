# ruff: noqa: C408
"""F0.3 baseline ladder and headroom (F0.3 H3.1-H3.5; F0.4 K4.1; criteria/f0_headroom.yaml).

Network level (weight space, all tiles of the 8 conv layers, 1000 CIFAR-10 test images, AdaBS from
256 further images): none, GDC, per-row (= AIHWKit 'per-column'), per-input-column, bi-scale, B8,
GDC+AdaBS, nominal-D2 timer, B1a (Joshi reference columns); oracle = as-programmed weights.
Readings of the plan (also in run extra): headroom = acc_prog - max(GDC, per-row, per-column,
GDC+AdaBS); H3.1 ordering = none < GDC < GDC+AdaBS per seed at 1 d and 1 yr under D2 (gain =
AdaBS - GDC at 1 d); sign-consistent = headroom > 0 for every seed at every counted time point;
H3.3 timer loss = acc(B1a) - acc(timer); mismatch factor = true D2 drift_scale / nominal 1.0;
H3.4 per-tile AUCs (a_common.watchdog_auc_rows); H3.5 recovery fraction = error reduction of the
best of {GDC, per-row, per-column, bi-scale (+bias), rank-2 probe B5 (k=2,s=2)} over error
reduction of M4best, pooled per (field, rho, time) at rho in {1, 2}, times {1 d, 30 d}, output error;
K4.1 t_star = first time (40-point log grid 60 s..10 yr) at which rms(W(t)-Wp) of a nominal D2
256-tile exceeds half a rail-to-rail stuck delta, taken as 0.5*max|w| (no GDC refresh).

AdaBS calibration (--adabs-cal): "train_aug" (default, fixed) = first 256 of a fixed pool of
augmented TRAIN images (matches checkpoint training); "test_clean" = legacy first run
(first 256 clean test images; biased, failed H3.1 sanity), kept only for provenance.

  python experiments/spectral/f0_3_headroom.py --jobs 6 --allow-dirty [--smoke] [--adabs-cal ...]
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from a_common import (
    ENC,
    PROG,
    TileCtx,
    make_drift,
    net_job,
    select_tiles,
    tile_conditions,
    watchdog_auc_rows,
)
from f0_common import finish, load_config, parse_args

from neurofault.spectral.harness import run_grid
from neurofault.spectral.types import make_generator

INC = ["acc_gdc", "acc_row", "acc_col", "acc_gdc_adabs"]


def tstar_task(point: dict) -> dict:
    seed, layer, r0, c0, m, n, size = point["tile"]
    ctx = TileCtx(seed, layer, r0, c0, m, n, size)
    real = make_drift("D2").sample(ctx.stp, make_generator(seed, f"drift:{layer}:{r0}:{c0}"))
    thr = 0.5 * float(ctx.wp.abs().max())
    ts = np.logspace(np.log10(60.0), np.log10(10 * 365.25 * 86400), 40)
    star = 1e9
    for t in ts:
        e = real.at(float(t)).weights() - ctx.wp
        if float(e.pow(2).mean().sqrt()) > thr:
            star = float(t)
            break
    wt = real.at(86400.0).weights() - ctx.wp
    return dict(kind="tstar", seed=seed, layer=layer, tile=f"{r0}_{c0}", model="D2",
                drift_scale=1.0, tile_size=size, t_star_s=star,
                rms_over_thr_1d=float(wt.pow(2).mean().sqrt()) / thr)  # fmt: skip


def net_metrics(net: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    net = net.copy()
    net["field"] = net["field"].fillna("")
    net["headroom_pt"] = net.acc_prog - net[INC].max(axis=1)
    net["headroom_b8_pt"] = net.acc_prog - net[[*INC, "acc_b8"]].max(axis=1)
    rows = []
    # H3.1
    d2 = net[(net.model == "D2") & (net.scale == 1.0) & net.time.isin(["1d", "1yr"])]
    for _, r in d2.iterrows():
        ok = int(r.acc_none < r.acc_gdc < r.acc_gdc_adabs)
        rows.append(dict(metric_group="H3.1", seed=r.seed, time=r.time, h31_ordering_ok=ok,
                         h31_gain_1d_pt=(r.acc_gdc_adabs - r.acc_gdc) if r.time == "1d" else np.nan,
                         acc_none=r.acc_none, acc_gdc=r.acc_gdc, acc_gdc_adabs=r.acc_gdc_adabs))  # fmt: skip
    # H3.2
    keys = ["model", "field", "rho", "scale"]
    for key, g in net.groupby(keys, dropna=False):
        per_t = g.groupby("time").headroom_pt.agg(["mean", "min"])
        per_t["ge1"] = per_t["mean"] >= 1.0
        consistent = bool(
            all((g[g.time == t].headroom_pt > 0).all() for t in per_t.index[per_t.ge1])
        )
        model, field, rho, scale = key
        elig = (
            (model == "D2" and scale in (1.0, 2.0))
            or (model == "D3" and rho <= 2.0)
            or model == "D4"
        )
        rows.append(dict(metric_group="H3.2", model=model, field=field, rho=rho, scale=scale,
                         h32_model_eligible=bool(elig),
                         h32_n_timepoints_ge_1pt=int(per_t.ge1.sum()),
                         h32_sign_consistent_across_seeds=consistent,
                         headroom_by_time=str(per_t["mean"].round(2).to_dict())))  # fmt: skip
    # H3.3
    t = net[net.model == "D2"].dropna(subset=["acc_timer", "acc_b1a"])
    for (scale, tm), g in t.groupby(["scale", "time"]):
        rows.append(dict(metric_group="H3.3", mismatch_factor=scale, time=tm,
                         timer_loss_vs_b1a_pt=float((g.acc_b1a - g.acc_timer).mean()),
                         acc_timer=g.acc_timer.mean(), acc_b1a=g.acc_b1a.mean(),
                         acc_gdc=g.acc_gdc.mean(), acc_prog=g.acc_prog.mean()))  # fmt: skip
    return pd.DataFrame(rows), {"net_with_headroom": net}


def h35_rows(cond: pd.DataFrame) -> pd.DataFrame:
    c = cond[
        cond.model.isin(["D3", "D4rho"]) & cond.rho.isin([1.0, 2.0]) & cond.time.isin(["1d", "30d"])
    ]
    p = c.groupby(["model", "field", "rho", "time"], dropna=False)[
        ["err_none", "err_b8", "err_b5s", "err_m4best", "err_m4cos", "err_m4r1"]].mean().reset_index()  # fmt: skip
    p["err_bestB"] = p[["err_b8", "err_b5s"]].min(axis=1)
    red_b, red_s = p.err_none - p.err_bestB, p.err_none - p.err_m4best
    p["b5_recovery_fraction_of_spectral"] = np.where(
        red_s > 1e-9, red_b / red_s.clip(lower=1e-9), 1.0
    )
    p["metric_group"] = "H3.5"
    return p


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    cal = "train_aug"
    if "--adabs-cal" in argv:
        i = argv.index("--adabs-cal")
        cal = argv[i + 1]
        del argv[i : i + 2]
    a = parse_args(__doc__.splitlines()[0], "f0_3.yaml", "f0_3", argv)
    cfg = load_config(a.config, a.smoke)
    cfg.update(master_seed=a.seed, enc=str(ENC), prog=str(PROG))
    out = Path(a.out)
    tiles = select_tiles(cfg["ckpt_seeds"], cfg["layers"], cfg["tiles_per_layer"])
    cond_rows = run_grid(tile_conditions, [dict(cfg=cfg, tile=t) for t in tiles], jobs=a.jobs,
                         out_dir=out / "_cells_cache")  # fmt: skip
    cond = cond_rows[cond_rows.kind == "cond"].dropna(axis=1, how="all")
    cond["field"] = cond["field"].fillna("")
    ts = run_grid(
        tstar_task, [dict(tile=t) for t in tiles], jobs=a.jobs, out_dir=out / "_cells_tstar"
    )
    net_pts = [
        dict(cfg=cfg, seed=s, scenario=sc, adabs_cal=cal)
        for s in cfg["ckpt_seeds"]
        for sc in cfg["net_scenarios"]
    ]
    net = run_grid(net_job, net_pts, jobs=a.jobs, out_dir=out / "_cells_net")
    nm, side = net_metrics(net)
    wa = watchdog_auc_rows(cond)
    h35 = h35_rows(cond)
    metrics = pd.concat(
        [
            nm,
            wa.assign(metric_group="H3.4"),
            h35,
            ts.drop(columns=["kind"]).assign(metric_group="K4.1"),
        ],
        ignore_index=True,
    )
    pd.set_option("display.width", 250)
    print("\n== network accuracy (mean over seeds) ==")
    print(side["net_with_headroom"].groupby(["model", "field", "rho", "scale", "time"], dropna=False).mean(numeric_only=True)
          .drop(columns=["seed"]).round(2).to_string())  # fmt: skip
    print(
        "\n== H3.2 ==\n",
        nm[nm.metric_group == "H3.2"].drop(columns="metric_group").to_string(index=False),
    )
    print(
        "\n== H3.3 ==\n",
        nm[nm.metric_group == "H3.3"].drop(columns="metric_group").round(2).to_string(index=False),
    )
    print("\n== H3.5 ==\n", h35.round(4).to_string(index=False))
    print("\nH3.4 median AUCs:", wa[["auc_sigma1_ratio", "auc_gdc_factor", "auc_current_sum",
                                      "auc_sigma1_minus_auc_gdc_factor"]].median().round(3).to_dict())  # fmt: skip
    print("K4.1 t_star median:", ts.t_star_s.median())
    finish("F0.3", "f0_headroom.yaml", cfg, a, metrics, extra={"note": "see module docstring", "adabs_cal": cal},
           side_tables={"net_with_headroom": side["net_with_headroom"], "tile_conditions": cond,
                        "watchdog_aucs": wa})  # fmt: skip
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
