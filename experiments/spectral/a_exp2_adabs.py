# ruff: noqa: C408
"""Direction A Exp 2 = master-plan arm P4.3 (C1' calibration-cost reduction vs AdaBS).

Network-level accuracy of {none, GDC, best cheap incumbent (B8), AdaBS (N in 64..2600, on GDC),
M4 field fit (probe budgets 8/16/32/64 MVM per tile), M4 + AdaBS-256, oracle} over simulated time
under D2 (iid) and D3/D4 structured drift, on W1 (ResNet-9) and W2 (BN folded into the weights,
AdaBS not applicable). Cost = counted calibration MVMs; AdaBS = images x 10240 tile-MVMs per image
(sum over layers of tiles x patch positions). Criteria: criteria/a_success.yaml (FROZEN).

Conventions (literal readings, all reported):
- B8 = per-tile best of {GDC, per-row, per-column, bi-scale} + mean-bias, chosen by held-out
  output error (selection oracle, optimistic for the incumbent); its cost = the winner's reads.
- AdaBS = GDC + BN re-estimation on TRAIN images with the training augmentation (see
  a_exp2_jobs docstring for why clean/test images are wrong for these checkpoints).
- All probe-based methods read through 2% per-cell analog read noise.
- Time points are 1h/1d/30d/1yr (plan's t/t0 grid is 1e4..1e8). D4 rows carry r = 1.0 (assumed
  thermal default has no amplitude axis). Devices: PCM only (S2 needs >= 2 devices).
- S1/K clauses name M2/M3; this arm evaluates M4 (the Gate-0' survivor), emitted under the same
  metric names, method = m4_B.

  python experiments/spectral/a_exp2_adabs.py --jobs 3 --allow-dirty [--smoke]
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from a_exp2_jobs import net_job, oracle_job
from f0_common import finish, load_config, parse_args

from neurofault.spectral.exp2 import cluster_bootstrap_ci, pareto_front
from neurofault.spectral.harness import run_grid

KEYS = ["workload", "seed", "sname", "time"]


def _budget(cfg_name: str) -> int | None:
    if cfg_name.startswith(("m4_", "m4ada_")):
        return int(cfg_name.split("_")[1])
    return None


def paired(df: pd.DataFrame, a: str, b: str, wl: str, scen=None) -> pd.DataFrame:
    """Rows (workload, seed, sname, time, model, field, rho, d=acc(a)-acc(b)) where both exist."""
    d = df[(df.workload == wl) & df.config.isin([a, b])]
    if scen is not None:
        d = d[d.sname == scen]
    idx = KEYS + ["model", "field", "rho", "device"]
    p = d.pivot_table(index=idx, columns="config", values="acc", dropna=False).reset_index()
    if a not in p or b not in p:
        return pd.DataFrame(columns=idx + ["d"])
    p = p.dropna(subset=[a, b])
    p["d"] = p[a] - p[b]
    return p


def s1_row(p: pd.DataFrame) -> dict:
    """S1 statistics of paired differences p.d over (seed, time) cells of one scenario."""
    sm = p.groupby("time").d.mean()
    m, lo, hi = cluster_bootstrap_ci(p.d.to_numpy(), p.seed.to_numpy(), seed=1)
    per_seed = p.groupby("seed").d.mean()
    return dict(
        s1_n_timepoints_gain_ge_1pt=int((sm >= 1.0).sum()), s1_ci_lower_paired_pt=lo,
        s1_ci_upper_paired_pt=hi, s1_mean_gain_pt=m,
        s1_sign_consistent_all_seeds=int((per_seed > 0).all()),
    )  # fmt: skip


def s1_met(r: dict) -> bool:
    return (
        r["s1_n_timepoints_gain_ge_1pt"] >= 2
        and r["s1_ci_lower_paired_pt"] > 0
        and r["s1_sign_consistent_all_seeds"] == 1
    )


def build_metrics(df: pd.DataFrame, orc: pd.DataFrame, cost: pd.DataFrame):
    rows, tabs = [], {}
    ratio = float(cost.loc[cost.config == "ada2600", "cost_mvm"].mean())
    cost_of = cost.set_index("config").cost_mvm.to_dict()
    struct = df[df.model.isin(["D3", "D4"])]
    scens = struct[["sname", "model", "field", "rho", "device"]].drop_duplicates()
    methods = sorted(c for c in df.config.unique() if c.startswith("m4_"))
    stacked = sorted(c for c in df.config.unique() if c.startswith("m4ada_"))
    s1_rows, k1_rows = [], []
    # ---- S1 (vs B8) and S1+ (stacked, vs b8_ada256 when present else gdc-ada256)
    for _, sc in scens.iterrows():
        for m in methods:
            p = paired(df, m, "b8", "W1", sc.sname)
            if not len(p):
                continue
            r = dict(kind="s1", method=m, workload="W1", model=sc.model, field=sc.field, r=sc.rho,
                     device=sc.device, mvm_budget_per_tile=_budget(m), stacked_adabs256=False,
                     scenario=sc.sname, **s1_row(p))  # fmt: skip
            s1_rows.append(r)
        for m in stacked:
            p = paired(df, m, "b8_ada256", "W1", sc.sname)
            ref = "b8_ada256"
            if not len(p):
                p, ref = paired(df, m, "ada256", "W1", sc.sname), "ada256(GDC)"
            if not len(p):
                continue
            s1_rows.append(dict(kind="s1plus", method=m, workload="W1", model=sc.model,
                                field=sc.field, r=sc.rho, device=sc.device,
                                mvm_budget_per_tile=_budget(m), stacked_adabs256=True,
                                scenario=sc.sname, ref=ref, **s1_row(p)))  # fmt: skip
    # D1/D2 sanity rows: delta vs B8 per (method, time)
    for m in methods:
        p = paired(df, m, "b8", "W1", "D2")
        for tl, g in p.groupby("time"):
            rows.append(dict(kind="d2_delta", method=m, workload="W1", model="D2", time=tl,
                             device="PCM", mvm_budget_per_tile=_budget(m),
                             delta_vs_b8_pt=g.d.mean()))  # fmt: skip
    # ---- S2 (vs AdaBS-2600), per method pooled over its W1 cells; per scenario in tables
    w1 = df[df.workload == "W1"]
    s2_cells = []
    for c in sorted(w1.config.unique()):
        if c == "ada2600":
            continue
        p = paired(df, c, "ada2600", "W1")
        if not len(p):
            continue
        cell = p.groupby(["sname", "time", "device"]).d.mean().reset_index()
        for dev, g in cell.groupby("device"):
            mean_abs = float(g.d.abs().mean())
            mean_signed, lo, hi = cluster_bootstrap_ci(
                p[p.device == dev].d.to_numpy(), p[p.device == dev].seed.to_numpy(), seed=2
            )
            rows.append(dict(kind="s2", method=c, workload="W1", device=dev,
                             abs_gap_vs_adabs2600_pt=mean_abs, signed_gap_vs_adabs2600_pt=mean_signed,
                             signed_gap_ci_lo=lo, signed_gap_ci_hi=hi,
                             worst_cell_gap_pt=float(g.d.abs().max()), n_cells=len(g),
                             cost_ratio_vs_adabs2600=cost_of[c] / ratio,
                             cost_mvm=cost_of[c]))  # fmt: skip
        for (sn, dev), g in cell.groupby(["sname", "device"]):
            s2_cells.append(dict(config=c, sname=sn, device=dev, abs_gap=float(g.d.abs().mean()),
                                 signed_gap=float(g.d.mean()),
                                 cost_ratio=cost_of[c] / ratio))  # fmt: skip
    s2_df = pd.DataFrame(s2_cells)
    tabs["s2_per_scenario"] = s2_df
    # ---- K1 rows
    for r in s1_rows:
        if r["kind"] != "s1":
            continue
        c = s2_df[(s2_df.config == r["method"]) & (s2_df.sname == r["scenario"])]
        s2_ok = bool(len(c) and c.abs_gap.iloc[0] <= 0.3 and c.cost_ratio.iloc[0] <= 0.1)
        k1_rows.append(dict(kind="k1", method=r["method"], model=r["model"], field=r["field"],
                            r=r["r"], device=r["device"], scenario=r["scenario"],
                            reaches_s1_or_s2=int(s1_met(r) or s2_ok), reaches_s1=int(s1_met(r)),
                            reaches_s2_row=int(s2_ok)))  # fmt: skip
    # ---- K2
    oc = orc[(orc.config == "oracle") & (orc.workload == "W1")].set_index("seed").acc
    ca = df[(df.workload == "W1") & (df.config == "col_ada256")].copy()
    if len(ca):
        ca["gap"] = ca.seed.map(oc) - ca.acc
        for (sn, tl), g in ca.groupby(["sname", "time"]):
            rows.append(dict(kind="k2_w1", workload="W1", scenario=sn, time=tl,
                             adabs256_percol_gap_to_oracle_pt=g.gap.mean()))  # fmt: skip
    for m in methods:
        p = paired(df, m, "b8", "W2")
        for (sn, tl), g in p.groupby(["sname", "time"]):
            rows.append(dict(kind="k2_w2", workload="W2", scenario=sn, time=tl, method=m,
                             corrector_gain_pt=g.d.mean()))  # fmt: skip
    # ---- K4
    fam = {}
    for r in s1_rows:
        if r["kind"] == "s1":
            fam[r["model"]] = fam.get(r["model"], False) or s1_met(r)
    d2min = min([r["delta_vs_b8_pt"] for r in rows if r["kind"] == "d2_delta"] or [np.nan])
    n_fam = sum(bool(v) and (d2min >= -0.3) for v in fam.values())
    rows.append(dict(kind="k4", n_families_meeting_s1=n_fam, families=str(fam), d2_min_delta=d2min))
    # S1+ (stacked_adabs256) rows are report-only and would leak into S1's unfiltered clauses
    rows += [r for r in s1_rows if r["kind"] == "s1"] + k1_rows
    tabs["s1_rows"] = pd.DataFrame(s1_rows)
    return pd.DataFrame(rows), tabs


def pareto_tables(df: pd.DataFrame, orc: pd.DataFrame) -> dict[str, pd.DataFrame]:
    out = {}
    for wl in ("W1", "W2"):
        d = df[df.workload == wl]
        if d.empty:
            continue
        full_scen = d[d.config == "m4_8"].sname.unique()  # scenarios with the full config set
        base = d[d.sname.isin(full_scen)]
        recs = []
        for c, g in base.groupby("config"):
            row = dict(config=c, cost_mvm=g.cost_mvm.mean(), acc=g.acc.mean(),
                       acc_seed_sd=g.groupby("seed").acc.mean().std())  # fmt: skip
            for tl, gt in g.groupby("time"):
                row[f"acc_{tl}"] = gt.acc.mean()
            for ref in ("b8", "ada2600", "gdc"):
                if ref == c:
                    continue
                p = paired(df[df.sname.isin(full_scen)], c, ref, wl)
                if len(p):
                    mm, lo, hi = cluster_bootstrap_ci(p.d.to_numpy(), p.seed.to_numpy(), seed=3)
                    row[f"d_vs_{ref}"], row[f"d_vs_{ref}_lo"], row[f"d_vs_{ref}_hi"] = mm, lo, hi
            recs.append(row)
        for c in ("oracle", "ref_digital"):
            o = orc[(orc.workload == wl) & (orc.config == c)]
            if len(o):
                recs.append(dict(config=c, cost_mvm=0, acc=o.acc.mean(),
                                 acc_seed_sd=o.acc.std()))  # fmt: skip
        t = pd.DataFrame(recs).sort_values("cost_mvm")
        pts = [(r.cost_mvm, r.acc) for r in t.itertuples()]
        t["on_pareto_front"] = False
        t.iloc[pareto_front(pts), t.columns.get_loc("on_pareto_front")] = True
        t["scenarios"] = ",".join(full_scen)
        out[f"pareto_{wl}"] = t
        # per-scenario/time long table (seed mean, sd)
        out[f"acc_by_cell_{wl}"] = (
            d.groupby(["sname", "model", "field", "rho", "time", "config"])
            .agg(acc=("acc", "mean"), acc_sd=("acc", "std"), cost_mvm=("cost_mvm", "mean"),
                 n_seeds=("seed", "nunique"))
            .reset_index()
        )  # fmt: skip
    return out


def _evaluate_device_code(crit, metrics):
    """verdict.evaluate casts the clause metric to float, so S2's `metric: device, nunique` (a
    string column) cannot be evaluated as frozen. In-memory copy of the criteria with that one
    metric renamed to the numeric twin `device_code` (PCM=1, RRAM=2); file untouched, verdict.json
    carries the note. Filters on `device: PCM` (S1) keep using the string column."""
    import copy

    from neurofault.spectral.verdict import evaluate

    def walk(n):
        if isinstance(n, dict):
            if n.get("metric") == "device" and n.get("aggregation") == "nunique":
                n["metric"] = "device_code"
            for v in n.values():
                walk(v)
        elif isinstance(n, list):
            for v in n:
                walk(v)

    c2 = copy.copy(crit)
    c2.criteria = copy.deepcopy(crit.criteria)
    walk(c2.criteria)
    v = evaluate(c2, metrics)
    v["note"] = "S2 metric 'device' (nunique) evaluated on numeric twin device_code; see script"
    return v


def _read_cache(d: Path) -> pd.DataFrame:
    rows = []
    for line in (d / "cells.jsonl").read_text().splitlines():
        if line.strip():
            rows += json.loads(line)["rows"]
    return pd.DataFrame(rows)


def main(argv=None) -> int:
    a = parse_args(__doc__.splitlines()[0], "a_exp2.yaml", "a_exp2", argv)
    cfg = load_config(a.config, a.smoke)
    cfg.update(master_seed=a.seed)
    out = Path(a.out)
    pts, opts = [], []
    for s in cfg["ckpt_seeds"]:
        for wl in ("W1", "W2"):
            opts.append(dict(cfg=cfg, seed=s, workload=wl))
            for sc in cfg["scenarios"]:
                if wl in sc["workloads"]:
                    pts.append(dict(cfg=cfg, seed=s, workload=wl, scenario=sc, sname=sc["name"],
                                    model=sc["model"], field=sc["field"], rho=sc["rho"],
                                    device=sc.get("device", "PCM")))  # fmt: skip
    # heaviest jobs first for load balance
    pts.sort(key=lambda p: -len(cfg["evalsets"][p["workload"]][p["scenario"]["tier"]]))
    if os.environ.get("A_EXP2_CACHE_ONLY"):  # analyse whatever cells are finished (no new jobs)
        orc, df = _read_cache(out / "_cells_oracle"), _read_cache(out / "_cells_net")
    else:
        orc = run_grid(oracle_job, opts, jobs=a.jobs, out_dir=out / "_cells_oracle")
        df = run_grid(net_job, pts, jobs=a.jobs, out_dir=out / "_cells_net")
    df["field"] = df["field"].fillna("")
    cost = df[df.workload == "W1"].groupby("config").cost_mvm.mean().reset_index()
    metrics, tabs = build_metrics(df, orc, cost)
    metrics["device_code"] = metrics["device"].map({"PCM": 1.0, "RRAM": 2.0})
    import f0_common

    f0_common.evaluate = _evaluate_device_code
    tabs.update(pareto_tables(df, orc))
    tabs["net_raw"] = df
    tabs["oracle_raw"] = orc
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    for k in ("pareto_W1", "pareto_W2"):
        if k in tabs:
            print(f"\n== {k} (acc %, cost in MVM-equivalents) ==")
            print(tabs[k].round(3).to_string(index=False))
    print("\n== S2 pooled rows ==")
    s2 = metrics[metrics.kind == "s2"].dropna(axis=1, how="all")
    print(s2.round(4).to_string(index=False))
    finish("A-Exp2", "a_success.yaml", cfg, a, metrics,
           extra={"note": "M4 arm of Exp 2; see script docstring for conventions"},
           side_tables=tabs)  # fmt: skip
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
