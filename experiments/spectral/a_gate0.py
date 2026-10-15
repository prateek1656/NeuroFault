# ruff: noqa: C408
"""Direction A Gate 0' (G0-a..f; criteria/a_gate0.yaml; arm P1.5).

Tile level (all 8 conv layers, tile 256, `tiles_per_layer` random tiles, 3 seeds): D3 fields x rho
and D4 (temperature-amplitude axis) x six times-of-structure ratio r in {0,.25,.5,1,2,4}, noise-free
and 2% per-cell probe read noise, 4 times. Network level (G0-d): all tiles, weight-space ResNet-9,
B8 vs B8 + oracle rank-16 correction. Metric rows are SEED MEANS; see `side tables`.

Interpretations (literal readings; see also a_common docstring):
  G0-a  excess = mean_tiles err(B8; r) - mean_tiles err(B8; r=0) in pp, noise-free reads.
  G0-b  capture(centred post-B8 residual, centred-covariance weighted) by M2 (oracle best rank-k)
        and M3 (oracle top-k centred PCs); gap = capture(r) - capture(D2, r=0).
  G0-c  realizable M2 (k=8, s=4, 2% noise) capture / oracle best-rank-8 capture, same residual.
  G0-e  per-tile partial AUC of |sigma1(t)/sigma1(0)-1| given quantile bins of |alpha_GDC-1|;
        label = B8 error above the tile's median B8 error (no network '1 pt' available per tile);
        exact SVD, not noisy power iteration (optimistic for M1).
  G0-f  Jacobian of [sigma1 ratio, sigma1/sigma2, v1 rotation] w.r.t. (log-time, aging=nu scale,
        temperature offset); rank tol 1e-6 of the column-normalised J.
D6 (G+/G- asymmetry) is not implemented in drift.py and is NOT run.

  python experiments/spectral/a_gate0.py --jobs 6 --allow-dirty [--smoke]
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
    coerce_numeric_strings,
    net_job,
    select_tiles,
    tile_conditions,
    watchdog_auc_rows,
)
from f0_common import finish, load_config, parse_args

from neurofault.spectral.harness import run_grid

KEY = ["model", "field", "rho"]


def _seed_mean(df: pd.DataFrame, by: list[str], cols: list[str]) -> pd.DataFrame:
    return df.groupby(by, dropna=False)[cols].mean().reset_index()


def tile_metrics(cond: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    cond = cond.copy()
    cond["field"] = cond["field"].fillna("")
    cond["model_key"] = cond["model"] + "|" + cond["field"]
    d2 = cond[(cond.model == "D2") & (cond.scale == 1.0)]
    err_cols = [c for c in cond.columns if c.startswith("err_")]
    cap_cols = [c for c in cond.columns if c.startswith("cap_")]
    # pooled over tiles per (seed, scenario, sigma, time)
    pooled = cond.groupby(["seed", "model", "field", "rho", "sigma", "time"], dropna=False)[
        err_cols + cap_cols
    ].mean().reset_index()  # fmt: skip
    base = pooled[(pooled.model == "D2")].set_index(["seed", "sigma", "time"])
    rows = []
    # G0-a (and the M4 analogue)
    for _, r in pooled[(pooled.model != "D2")].iterrows():
        b = base.loc[(r.seed, r.sigma, r.time)]
        rows.append(dict(
            metric_group="G0-a", seed=r.seed, model=r.model, field=r.field, r=r.rho, sigma=r.sigma,
            time=r.time, b8_excess_err_pp=100 * (r.err_b8 - b.err_b8),
            m4_excess_err_pp=100 * (r.err_m4best - b.err_m4best),
            err_b8_pp=100 * r.err_b8, err_m4best_pp=100 * r.err_m4best))  # fmt: skip
    ga = pd.DataFrame(rows)
    ga0 = ga[ga.sigma == 0.0].drop(columns="sigma")
    # G0-b
    brows = []
    for k in cfg["cap_ks"]:
        for meth in ("m2", "m3"):
            col = f"cap_{meth}_k{k}"
            for _, r in pooled[(pooled.model != "D2") & (pooled.sigma == 0.0)].iterrows():
                b = base.loc[(r.seed, 0.0, r.time)]
                brows.append(dict(metric_group="G0-b", seed=r.seed, model=r.model, field=r.field,
                                  r=r.rho, time=r.time, k=k, method=meth.upper(),
                                  capture=r[col], capture_r0=b[col],
                                  specificity_gap=r[col] - b[col]))  # fmt: skip
    gb = pd.DataFrame(brows)
    # G0-c
    n = pooled[(pooled.sigma == 0.02) & (pooled.model != "D2")]
    gc = n.assign(
        metric_group="G0-c", r=n.rho,
        realizable_over_oracle_capture=n.cap_m2_real_k8 / n.cap_m2_k8.clip(lower=1e-12),
    )[["metric_group", "seed", "model", "field", "r", "time", "realizable_over_oracle_capture",
       "cap_m2_real_k8", "cap_m2_k8"]]  # fmt: skip
    # G0-e/f
    wa = watchdog_auc_rows(cond)
    ge = wa.assign(metric_group="G0-e")
    jac = pd.DataFrame(cond.attrs.get("jac_rows", []))
    side = {"tile_conditions": cond, "pooled": pooled, "m1_aucs": wa}
    parts = [ga0, gb, gc, ge]
    if len(jac):
        parts.append(jac.assign(metric_group="G0-f"))
    # crossover table: incumbent vs M4 by field / rho / noise (mean over seeds, times)
    cross = (
        pooled[pooled.model != "D2"]
        .groupby(["model", "field", "rho", "sigma"], dropna=False)[
            [
                "err_b8",
                "err_m4best",
                "err_m4cos",
                "err_m4r1",
                "err_gdc",
                "err_row",
                "err_col",
                "err_bi",
                "err_m3",
                "err_b5",
            ]
        ]
        .mean()
        .reset_index()
    )
    cross["m4_gain_pp"] = 100 * (cross.err_b8 - cross.err_m4best)
    side["incumbent_vs_m4"] = cross
    _ = d2
    return pd.concat(parts, ignore_index=True), side


def net_metrics(net: pd.DataFrame) -> pd.DataFrame:
    net = net.copy()
    net["gap"] = net.acc_b8o16 - net.acc_b8
    g = net.groupby(["model", "field", "rho", "t_s"], dropna=False).agg(
        oracle16_vs_b8_acc_gap_pt=("gap", "mean"), gap_min_seed=("gap", "min"),
        gap_max_seed=("gap", "max"), acc_b8=("acc_b8", "mean"), acc_b8o16=("acc_b8o16", "mean"),
        acc_none=("acc_none", "mean"), acc_prog=("acc_prog", "mean")).reset_index()  # fmt: skip
    g["r"] = g.rho
    g["metric_group"] = "G0-d"
    return g


def main(argv=None) -> int:
    a = parse_args(__doc__.splitlines()[0], "a_gate0.yaml", "a_gate0", argv)
    cfg = load_config(a.config, a.smoke)
    cfg.update(master_seed=a.seed, enc=str(ENC), prog=str(PROG))
    out = Path(a.out)
    tiles = select_tiles(cfg["ckpt_seeds"], cfg["layers"], cfg["tiles_per_layer"])
    cond_rows = run_grid(tile_conditions, [dict(cfg=cfg, tile=t) for t in tiles], jobs=a.jobs,
                         out_dir=out / "_cells_cache")  # fmt: skip
    jac = cond_rows[cond_rows.get("kind") == "jac"].dropna(axis=1, how="all")
    cond = cond_rows[cond_rows.get("kind") == "cond"].dropna(axis=1, how="all")
    cond.attrs["jac_rows"] = jac.drop(columns=["kind"]).to_dict("records")
    net_pts = [dict(cfg=cfg, seed=s, scenario=sc) for s in cfg["ckpt_seeds"]
               for sc in cfg["net_scenarios"]]  # fmt: skip
    net = run_grid(net_job, net_pts, jobs=a.jobs, out_dir=out / "_cells_cache_net")
    tm, side = tile_metrics(cond, cfg)
    nm = net_metrics(net)
    metrics = pd.concat([tm, nm], ignore_index=True)
    side["net_raw"] = net
    # seed-mean over G0-a/b/c rows for the verdict (per-seed rows stay in side tables)
    side["per_seed_metrics"] = metrics
    sm = []
    for grp, d in metrics.groupby("metric_group"):
        if grp in ("G0-a", "G0-b", "G0-c"):
            keys = [c for c in ("model", "field", "r", "time", "k", "method") if c in d.columns]
            num = [c for c in d.columns if d[c].dtype.kind == "f" and c not in keys]
            sm.append(
                d.groupby(keys, dropna=False)[num].mean().reset_index().assign(metric_group=grp)
            )
        else:
            sm.append(d)
    verdict_metrics = pd.concat(sm, ignore_index=True)
    pd.set_option("display.width", 250)
    cross = side["incumbent_vs_m4"]
    print(
        "\n== incumbent (B8, best-of GDC/row/col/biscale + bias) vs M4best, output error (frac), mean over seeds/tiles/times =="
    )
    print(cross.round(4).to_string(index=False))
    for sg in cfg["sigmas"]:
        c = cross[cross.sigma == sg]
        for key, d in c.groupby(["model", "field"], dropna=False):
            win = d[d.m4_gain_pp > 0].rho.min() if (d.m4_gain_pp > 0).any() else np.nan
            win1 = d[d.m4_gain_pp >= 1.0].rho.min() if (d.m4_gain_pp >= 1.0).any() else np.nan
            print(f"  rho* sigma={sg} {key}: M4 beats B8 at rho>={win}; by >=1pp at rho>={win1}")
    run_dir, _ = finish("A-Gate0", "a_gate0.yaml", cfg, a, verdict_metrics,
           extra={"note": "metric rows are seed means; per-seed in side tables"},
           side_tables={k: v for k, v in side.items() if hasattr(v, "to_csv")})  # fmt: skip
    # The frozen YAML writes t_s thresholds as '1.0e4' / '1.0e8', which PyYAML loads as strings, so
    # G0-d / GK-* clauses that filter on t_s evaluate to 'no data' above. Same criteria, numerics
    # coerced in memory only (file untouched), as a clearly separate second verdict.
    import copy

    from f0_common import CRITERIA_DIR

    from neurofault.spectral.verdict import evaluate, load_criteria, write_verdict

    crit = load_criteria(CRITERIA_DIR / "a_gate0.yaml")
    crit2 = copy.copy(crit)
    crit2.criteria = coerce_numeric_strings(crit.criteria)
    v2 = evaluate(crit2, verdict_metrics)
    write_verdict(
        v2, run_dir / "verdict_numeric_coerced.json", extra={"note": "t_s thresholds coerced"}
    )
    print(f"\n=== A-Gate0 verdict with t_s thresholds coerced to float: {v2['overall']} ===")
    for c in v2["criteria"]:
        print(f"  {c['id']:<18s} {c['kind']:<8s} {c['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
