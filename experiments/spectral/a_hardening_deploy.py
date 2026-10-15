# ruff: noqa: C408
"""Phase H / H2b: DEPLOYABLE M4 vs the H2 baselines, paired with a_hardening (criteria/a_hardening_deploy.yaml).

Rebuilds exactly the a_hardening tiles, drift realizations and M4 probe/bias streams (+2000), recomputes the H2 M4
(ideal, digital-reference form) as a pairing check, and evaluates the deployable G-forms M4D_full (rank <= 4) and
M4D_r1 (rank 1). Baseline accuracies (B8, HB1_N8, HB1_N32, HB2, HB3) are read from the final a_hardening run dir.

  python experiments/spectral/a_hardening_deploy.py --h2-run results/spectral/a_hardening/<run> --jobs 3 --allow-dirty
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import a_hardening as ah
import numpy as np
import pandas as pd
import torch
from a_bnfold import _tiles
from a_common import ENC, PROG, TileCtx, make_drift
from f0_common import finish, load_config, parse_args

from neurofault.spectral.encoding import encode
from neurofault.spectral.hardening import interleave_index, merge_rows, reference_weights, take_rows
from neurofault.spectral.harness import run_grid
from neurofault.spectral.methods_a import deployable_correction
from neurofault.spectral.noise import program
from neurofault.spectral.types import Tile, make_generator

BASES = ("B8", "HB1_N8", "HB1_N32", "HB2", "HB3")


class MRun(ah.HRun):
    def at(self, t: float) -> dict:
        ctx = self.ctx
        aug = self.real_aug.at(t)
        self.cur_aug[0], self.cur[0] = aug, take_rows(aug, self.arr_idx)
        wt = self.cur[0].weights()
        c = self.m4.update(self.reader2, t)
        x2 = self.reader2.mvm(ctx.xbar[:, None])[:, 0]
        out = {"m4": self._with_bias(c, x2, wt, self.y0bar)}
        for name, rk in (("m4d_full", None), ("m4d_r1", 1)):
            w, terms = deployable_correction(wt, c.meta["F"], 4, rk)
            out[name] = dict(W=w, b=self.y0bar - (x2 + (w - wt) @ ctx.xbar), passes=terms - 1)
        return out


def net_job(p: dict) -> list[dict]:
    torch.set_num_threads(1)
    cfg, seed, ds, sc = p["cfg"], p["seed"], p["dataset"], p["scenario"]
    h2 = cfg["hb2"]
    bridge = ah.HBridge(ds, seed, cfg["n_eval"], p["device"])
    saved = ah._install(ds, seed, bridge)
    rseed = seed + ah.SEED_OFFSET
    try:
        mats = bridge.layer_mats()
        mod = make_drift(sc["model"], sc.get("field") or "row_gradient", sc["rho"], 1.0)
        runs = []
        for layer, r0, c0, m, n in _tiles(bridge):
            ctx = TileCtx(seed, layer, r0, c0, m, n, 256)
            arr_idx, ref_idx, ref_pos = interleave_index(m, h2["k_ref"])
            absmax = float(ctx.tile.w.abs().max())
            wref = reference_weights(
                ctx.tile.w, h2["k_ref"], n, make_generator(rseed, f"hb2ref:{ctx.key}")
            )
            ref0 = encode(Tile(ctx.tile.spec, wref), ENC, absmax=absmax)
            ref_stp = program(ref0, PROG, make_generator(rseed, f"hb2prog:{ctx.key}"))
            aug0 = merge_rows(ctx.stp, ref_stp, arr_idx, ref_idx)
            real_aug = mod.sample(aug0, make_generator(rseed, f"drift:{layer}:{r0}:{c0}"))
            runs.append(((layer, r0, c0, m, n), MRun(ctx.slim(), real_aug, aug0, (arr_idx, ref_idx, ref_pos),
                                                     cfg["sigma"], rseed, h2, None)))  # fmt: skip
        rows = []
        for tlabel, t in cfg["times"].items():
            names = ("m4", "m4d_full", "m4d_r1")
            W = {k: {} for k in names}
            B = {k: {} for k in names}
            passes = {"m4d_full": [], "m4d_r1": []}
            for (layer, r0, c0, m, n), run in runs:
                res = run.at(t)
                for name in names:
                    W[name].setdefault(layer, torch.zeros_like(mats[layer], dtype=torch.float32))
                    W[name][layer][r0 : r0 + m, c0 : c0 + n] = res[name]["W"].float()
                    bd = B[name].setdefault(layer, torch.zeros(mats[layer].shape[0]))
                    bd[r0 : r0 + m] += res[name]["b"].float()
                for k, plist in passes.items():
                    plist.append(res[k]["passes"])
            acc = {k: bridge.acc(W[k], B[k]) for k in names}
            rows.append(dict(dataset=ds, ckpt_seed=seed, scenario=sc["name"], model=sc["model"], time=tlabel,
                             t_s=t, acc_m4_recomputed=acc["m4"], acc_m4d_full=acc["m4d_full"],
                             acc_m4d_r1=acc["m4d_r1"], extra_passes_m4d_full=max(passes["m4d_full"]),
                             extra_passes_m4d_full_mean=float(np.mean(passes["m4d_full"])),
                             extra_passes_m4d_r1=max(passes["m4d_r1"])))  # fmt: skip
            del W, B
        return rows
    finally:
        ah._restore(saved)


def build_metrics(net: pd.DataFrame, h2: pd.DataFrame) -> pd.DataFrame:
    key = ["dataset", "ckpt_seed", "scenario", "t_s"]
    h2 = h2[h2.method.isin(("M4",) + BASES)]
    m4_h2 = h2[h2.method == "M4"].set_index(key)["acc"].rename("acc_m4_h2")
    d = net.set_index(key).join(m4_h2)
    d["pairing_abs_diff_pt"] = (d.acc_m4_recomputed - d.acc_m4_h2).abs()
    base = h2[h2.method.isin(BASES)][key + ["method", "acc", "cost_mvm_eq"]]
    df = base.merge(d.reset_index(), on=key, how="left")
    for b in BASES:
        is_b = df.method == b
        df[f"m4d_minus_{b}_pt"] = np.where(is_b, df.acc_m4d_full - df.acc, np.nan)
        df[f"m4r1_minus_{b}_pt"] = np.where(is_b, df.acc_m4d_r1 - df.acc, np.nan)
    return df


def main(argv=None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    i = args.index("--h2-run")
    h2_run = Path(args[i + 1])
    del args[i : i + 2]
    a = parse_args(__doc__.splitlines()[0], "a_hardening.yaml", "a_hardening_deploy", args)
    cfg = load_config(a.config, a.smoke)
    cfg.update(master_seed=a.seed, eval_device=ah._EVAL_DEVICE)
    out = Path(a.out)
    h2 = pd.read_csv(h2_run / "metrics.csv")
    pts = [dict(cfg=cfg, seed=s, dataset=d, scenario=sc, device=ah._EVAL_DEVICE)
           for sc in cfg["scenarios"] for d in cfg["datasets"] for s in cfg["ckpt_seeds"]]  # fmt: skip
    net = run_grid(net_job, pts, jobs=a.jobs, out_dir=out / "_cells_net")
    df = build_metrics(net, h2)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    summ = net.groupby(["dataset", "t_s"])[
        ["acc_m4_recomputed", "acc_m4d_full", "acc_m4d_r1"]
    ].mean()
    summ = summ.join(h2.pivot_table(index=["dataset", "t_s"], columns="method", values="acc"))
    print(summ.round(2).to_string())
    print("max pairing diff (pt):", float(df.pairing_abs_diff_pt.max()))
    run_dir, _ = finish("A-hardening H2b", "a_hardening_deploy.yaml", cfg, a, df,
                        extra={"note": "H2b deployable M4 vs H2 baselines; paired with " + str(h2_run)},
                        side_tables=dict(summary=summ.reset_index(), net_raw=net))  # fmt: skip
    print(f"run dir: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
