"""E4 (criteria/e4_xabft.yaml): X-ABFT (Liu et al., TODAES 2020) as a baseline in the H8 comparison.

Same instances as H8 (experiments/spectral/b_vs_prior.py): ResNet-9 CIFAR-10 conv tiles of 256,
ckpts 42/101/102, offset_refcol, stuck_random mixed/pos, truth = visible faults, per-cell read noise
{1,2,4}% of max|w|, k in {1,4,16,64}, calibration = tile origin 1 with fault seeds 3000-3019,
evaluation = tile origin 0 with fault seeds 60-79. The H8 families (HPE, ROTH, CSF, CSB) are NOT
recomputed: their per-instance reads/F1 are joined from the registered H8 run's eval_instances.csv
(read only) so every comparison is paired on identical instances.

X-ABFT arms: ``sel`` = design chosen by the H8 selection rule (fewest calibration-mean reads with
calibration cell F1 >= 0.93, fallback highest F1) from the grid r_t in {4,8,16} x M in {2,4}
x rep in {1,4,16} (c_t = 16, linear f); ``paper`` = the paper's example design r_t=4, c_t=16, M=4, linear f, no
selection. The grid also has repeat-and-average factor rep in {1,4,16} (reads x rep, noise /sqrt(rep)) so X-ABFT
can buy SNR the way the other families can. Cost is the SAME read unit as H8 (counted forward MVMs); native cost columns:
n_test_vectors (= M ceil(n/r_t)), time_redundancy TR = (10000 + n_tv)/10000, hardware_redundancy
HR = 1 + 5/c_t. The checksum lines are digital/noise-free (favours X-ABFT; see xabft.py).

    uv run python experiments/spectral/e4_xabft.py [--smoke] [--allow-dirty] [--jobs 3] [--out DIR]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import b_c1_prereg as C
import b_e1 as B

from neurofault.spectral.harness import run_grid, write_run
from neurofault.spectral.localize import XABFTLocalizer
from neurofault.spectral.metrics import localization_f1, paired_ratio_ci
from neurofault.spectral.readout import TileReader
from neurofault.spectral.types import ReadNoise, make_generator

KS = [1, 4, 16, 64]
SIGMAS = [0.01, 0.02, 0.04]
TILE = 256
TARGET_CAL = 0.93
CAL_SEEDS, EV_SEEDS = (3000, 3020), (60, 80)
GRID = {f"r{r}M{m}R{q}": {"r_t": r, "c_t": 16, "M": m, "rep": q}
        for r in (4, 8, 16) for m in (2, 4) for q in (1, 4, 16)}  # fmt: skip
PAPER = {"r_t": 4, "c_t": 16, "M": 4, "rep": 1}
H8_DIR = B.REPO / "results" / "spectral" / "b_vs_prior"
KEYS = ["ckpt_seed", "layer", "fault_seed", "k", "sigma_c"]


class _AvgReader:
    """Repeat every forward read ``rep`` times and average: buys SNR with counted reads (X-ABFT's own
    paper has no such knob; offering it keeps the comparison fair in the noisy regime)."""

    def __init__(self, inner, rep):
        self.inner, self.rep = inner, rep
        self.backward_supported, self.shape = inner.backward_supported, inner.shape

    @property
    def cost(self):
        return self.inner.cost

    def mvm(self, x):
        return sum(self.inner.mvm(x) for _ in range(self.rep)) / self.rep


def _one(point, tile, healthy, bad, vis, sg, cfg):
    cfg = dict(cfg)
    rep = cfg.pop("rep", 1)
    noise = ReadNoise(mode="per_cell", sigma_cell_frac=sg, adc_bits=None, dac_bits=None)
    seed = B._seed(point) + 7919 * point["k"]
    reader = TileReader(lambda: cur[0], noise, make_generator(seed, f"read_xabft_{cfg}"),
                        backward_supported=False)  # fmt: skip
    cur = [healthy]  # closure above reads the cell at call time
    reader = _AvgReader(reader, rep) if rep > 1 else reader
    loc = XABFTLocalizer(tile.w, seed=seed, fa=point["fa"], **cfg)
    loc.calibrate(reader)
    cur[0] = bad
    est, cost = loc.localize(reader)
    return cost.mvm_fwd + cost.mvm_bwd, localization_f1(est.cells, vis), est.meta


def run_cal(point: dict) -> list[dict]:
    torch.set_num_threads(1)
    tile, healthy = B._tile_state(point)
    bad, vis = B._inject(point, healthy, tile)
    rows = []
    for sg in point["sigmas"]:
        res = [_one(point, tile, healthy, bad, vis, sg, c) for c in GRID.values()]
        rows.append({"sigma_c": sg, "n_visible": len(vis),
                     "reads": json.dumps([r[0] for r in res]),
                     "f1": json.dumps([round(r[1], 5) for r in res])})  # fmt: skip
    return rows


def select(cal: pd.DataFrame) -> dict:
    names, sel = list(GRID), {}
    for (k, sg), g in cal.groupby(["k", "sigma_c"]):
        R = np.array([json.loads(s) for s in g.reads], float).mean(0)
        F = np.array([json.loads(s) for s in g.f1], float).mean(0)
        idx = np.arange(len(names))
        ok = idx[F >= TARGET_CAL]
        if len(ok):
            j = ok[np.argmin(R[ok])]
        else:
            cand = idx[F >= F.max() - 1e-12]
            j = cand[np.argmin(R[cand])]
        sel[f"{int(k)}|{sg}"] = {"cfg": names[j], "cal_reads": float(R[j]), "cal_f1": float(F[j]),
                                 "cal_reached_target": bool(F[j] >= TARGET_CAL)}  # fmt: skip
    return sel


def run_eval(point: dict) -> list[dict]:
    torch.set_num_threads(1)
    tile, healthy = B._tile_state(point)
    bad, vis = B._inject(point, healthy, tile)
    rows = []
    for sg in point["sigmas"]:
        row = {"sigma_c": sg, "n_visible": len(vis)}
        name = point["sel"][f"{point['k']}|{sg}"]["cfg"]
        for tag, cfg in (("xabft", GRID[name]), ("xpaper", PAPER)):
            r, f, meta = _one(point, tile, healthy, bad, vis, sg, cfg)
            row[f"reads_{tag}"], row[f"f1_{tag}"] = r, f
            row[f"tr_{tag}"], row[f"hr_{tag}"] = (
                meta["time_redundancy"],
                meta["hardware_redundancy"],
            )
            row[f"undecodable_{tag}"], row[f"flagged_{tag}"] = (
                meta["undecodable"],
                meta["n_flagged"],
            )
        rows.append(row)
    return rows


def _row_ok(f1_cs, f1_x, ratio, hi, bar, bar_hi):
    if f1_cs < 0.9:
        return 0
    return int(f1_x < 0.9 or (ratio <= bar and hi <= bar_hi))


def summarize(ev: pd.DataFrame, h8: pd.DataFrame, sel: dict) -> pd.DataFrame:
    h8cols = KEYS + [c for c in h8.columns if c.startswith(("reads_", "f1_")) and c[-3:] != "ext"]
    m = ev.merge(h8[h8cols], on=KEYS, how="inner", suffixes=("", "_h8"))
    if len(m) != len(ev):
        raise RuntimeError(f"H8 join lost instances: {len(ev)} -> {len(m)}")
    rows = []
    for (k, sg), g in m.groupby(["k", "sigma_c"]):
        r = {"k": int(k), "sigma_c": float(sg), "n_instances": len(g)}
        for fam in ("hpe", "roth", "csf", "csb", "xabft", "xpaper"):
            r[f"reads_{fam}"], r[f"f1_{fam}"] = (
                float(g[f"reads_{fam}"].mean()),
                float(g[f"f1_{fam}"].mean()),
            )
        r["cfg_xabft"] = sel[f"{int(k)}|{sg}"]["cfg"]
        for tag in ("xabft", "xpaper"):
            r[f"tr_{tag}"], r[f"hr_{tag}"] = (
                float(g[f"tr_{tag}"].mean()),
                float(g[f"hr_{tag}"].mean()),
            )
            r[f"undecodable_frac_{tag}"] = float((g[f"undecodable_{tag}"] > 0).mean())
            r[f"{tag}_reaches"] = int(r[f"f1_{tag}"] >= 0.9)
            r[f"f1_ge90_frac_{tag}"] = float((g[f"f1_{tag}"] >= 0.9).mean())
        for cs in ("csb", "csf"):
            for x in ("xabft", "xpaper"):
                _lo, hi, ratio = paired_ratio_ci(g[f"reads_{cs}"].to_numpy(float),
                                                g[f"reads_{x}"].to_numpy(float), n_boot=4000)  # fmt: skip
                r[f"ratio_{cs}_{x}"], r[f"ratio_{cs}_{x}_ci_upper"] = ratio, hi
        for cs, x, name in (("csb", "xabft", "row_ok_2x_x"), ("csf", "xabft", "row_ok_fwd_2x_x"),
                            ("csb", "xpaper", "row_ok_2x_xpaper")):  # fmt: skip
            r[name] = _row_ok(r[f"f1_{cs}"], r[f"f1_{x}"], r[f"ratio_{cs}_{x}"],
                              r[f"ratio_{cs}_{x}_ci_upper"], 0.5, 0.6)  # fmt: skip
        # strict variant: X-ABFT must itself reach F1 0.9 (no credit for its failures)
        r["row_ok_2x_x_strict"] = int(r["f1_csb"] >= 0.9 and r["xabft_reaches"] == 1
                                      and r["ratio_csb_xabft"] <= 0.5
                                      and r["ratio_csb_xabft_ci_upper"] <= 0.6)  # fmt: skip
        r["xabft_cheaper_than_csb"] = int(
            r["xabft_reaches"] == 1 and r["reads_xabft"] < r["reads_csb"]
        )
        rows.append(r)
    return pd.DataFrame(rows).sort_values(["sigma_c", "k"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--allow-dirty", action="store_true")
    ap.add_argument("--jobs", type=int, default=3)
    ap.add_argument("--out", default=None)
    ap.add_argument("--h8-dir", default=str(H8_DIR))
    a = ap.parse_args()
    cal_seeds = range(CAL_SEEDS[0], CAL_SEEDS[0] + (2 if a.smoke else 20))
    ev_seeds = range(EV_SEEDS[0], EV_SEEDS[0] + (2 if a.smoke else 20))
    ckpts = [42] if a.smoke else C.CKPTS
    ks = [1, 16] if a.smoke else KS
    tag = "smoke" if a.smoke else "full"
    out = Path(
        a.out or B.REPO / "results" / "spectral" / ("e4_xabft" + ("_smoke" if a.smoke else ""))
    )
    h8_runs = sorted(Path(a.h8_dir).glob("2026*"))
    h8_csv = h8_runs[-1] / "eval_instances.csv"
    h8 = pd.read_csv(h8_csv)
    cal_pts = C.points("cal", TILE, cal_seeds, ks, SIGMAS, ckpts)
    ev_pts = C.points("eval", TILE, ev_seeds, ks, SIGMAS, ckpts)
    if a.smoke:
        cal_pts, ev_pts = cal_pts[: 2 * len(ks) * 2], ev_pts[: 2 * len(ks) * 2]
    print(f"calibration {len(cal_pts)} points, evaluation {len(ev_pts)} points", flush=True)
    cal = run_grid(run_cal, cal_pts, jobs=a.jobs, out_dir=out / f"_cells_cal_{tag}")
    sel = select(cal)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"selected_configs_{tag}.json").write_text(json.dumps(sel, indent=1, sort_keys=True))
    for p in ev_pts:
        p["sel"] = sel
    ev = run_grid(run_eval, ev_pts, jobs=a.jobs, out_dir=out / f"_cells_eval_{tag}")
    vm = summarize(ev, h8, sel)
    pd.set_option("display.width", 250, "display.max_columns", 60)
    cols = ["sigma_c", "k", "n_instances", "reads_roth", "f1_roth", "reads_csb", "f1_csb",
            "cfg_xabft", "reads_xabft", "f1_xabft", "reads_xpaper", "f1_xpaper",
            "undecodable_frac_xpaper", "ratio_csb_xabft", "row_ok_2x_x", "row_ok_2x_x_strict",
            "xabft_cheaper_than_csb"]  # fmt: skip
    print(vm[cols].round(3).to_string(index=False))
    cfg = {"ks": ks, "sigmas": SIGMAS, "tile": TILE, "cal_seeds": list(CAL_SEEDS),
           "eval_seeds": list(EV_SEEDS), "ckpts": ckpts, "grid": GRID, "paper": PAPER,
           "cal_target": TARGET_CAL, "h8_eval_csv": str(h8_csv), "selected": sel}  # fmt: skip
    rd = write_run(out, cfg, 0, vm, {"target_f1": TARGET_CAL}, extra={"mode": tag},
                   allow_dirty=a.allow_dirty, run_id=None)  # fmt: skip
    vm.to_csv(rd / "verdict_metrics.csv", index=False)
    ev.to_csv(rd / "eval_instances.csv", index=False)
    print("wrote", rd)


if __name__ == "__main__":
    main()
