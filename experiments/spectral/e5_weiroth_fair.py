"""E5 (criteria/e5_weiroth_fair.yaml): Wei & Roth (arXiv 2402.13503) on their own terms.

H8 compared our CS localizer with OUR cell-level reimplementation of Wei & Roth under Gaussian read
noise and a counted-reads axis. Wei & Roth's paper has bounded noise, stops at output LINES, and pays
r redundant columns inside ONE matrix-vector product. This script reruns the comparison on the SAME
instances as H8 (ResNet-9 CIFAR-10 conv tiles of 256, ckpts 42/101/102, offset_refcol, stuck_random,
truth = visible faults, sigma_c {1,2,4}% of max|w|, k {1,4,16,64}, evaluation tile origin 0, fault seeds
60-79) under two noise models:
  bounded : BoundedNoiseReader, +-delta iid sign noise with delta = sigma_c max|w| per unit input norm
            (rms-matched to the Gaussian model; the paper's model, bound known to the decoder);
  gauss   : our TileReader white read noise (H8's setting).
Arms (Wei & Roth code H = Kautz-Singleton, d_lines by k below; configuration FROZEN here, no tuning):
  wr_line_worst : paper decoder, threshold rho*delta (worst case), line level, 1 MVM + r columns;
  wr_line_rss   : same, random-noise threshold z*delta*sqrt(w_i) (not the paper's adversarial bound);
  wr_cell_worst / wr_cell_rss : line stage + OUR cell-level Kautz-Singleton extension (not in the paper);
  csf           : our forward-only CS decoder at the config H8 froze for (k, sigma) (H8 selection file).
Under gauss the 'worst' arms use bound 3 sigma (the Gaussian has no bound; 3 sigma is the usual proxy).
Metrics per instance: line_f1 (predicted output lines vs lines holding a visible fault), cell_f1,
reads (counted forward MVMs, H8 unit), r_cols (redundant columns in the single protected MVM; 0 for CS),
col_equiv = r_cols + (reads - 1) (the paper's cost with every extra probe read charged one column).
For gauss the H8 families (ROTH, CSB) are joined from the H8 eval_instances.csv (read only).

    uv run python experiments/spectral/e5_weiroth_fair.py [--smoke] [--allow-dirty] [--jobs 3] [--out DIR]
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
from neurofault.spectral.localize import BoundedNoiseReader, WeiRothPaperLocalizer
from neurofault.spectral.localize.weiroth import line_f1
from neurofault.spectral.metrics import localization_f1, paired_ratio_ci
from neurofault.spectral.readout import TileReader
from neurofault.spectral.types import ReadNoise, make_generator

KS = [1, 4, 16, 64]
SIGMAS = [0.01, 0.02, 0.04]
TILE = 256
EV_SEEDS = (60, 80)
D_LINES = {
    1: 1,
    4: 2,
    16: 4,
    64: 4,
}  # outlier lines the code is built for (r = 25/49/121/121 of 256)
H8_DIR = B.REPO / "results" / "spectral" / "b_vs_prior"
KEYS = ["ckpt_seed", "layer", "fault_seed", "k", "sigma_c"]
REP = 16  # repeat-and-average factor for the *_R arms (reads x REP, rms noise / sqrt(REP))
ARMS = {  # name: (cell_stage, threshold, repeats)
    "wr_line_worst": ("none", "worst", 1),
    "wr_line_rss": ("none", "rss", 1),
    "wr_line_rss_R": ("none", "rss", REP),
    "wr_cell_worst": ("ks", "worst", 1),
    "wr_cell_rss": ("ks", "rss", 1),
    "wr_cell_rss_R": ("ks", "rss", REP),
}
CELL_ARMS = ["wr_cell_worst", "wr_cell_rss", "wr_cell_rss_R"]


class _AvgReader:
    """Repeat every forward read ``rep`` times and average (reads counted x rep)."""

    def __init__(self, inner, rep):
        self.inner, self.rep = inner, rep
        self.backward_supported, self.shape = inner.backward_supported, inner.shape

    @property
    def cost(self):
        return self.inner.cost

    def mvm(self, x):
        return sum(self.inner.mvm(x) for _ in range(self.rep)) / self.rep


def _reader(noise_model, healthy_cell, wmax, sg, seed, tag):
    if noise_model == "bounded":
        inner = TileReader(lambda: healthy_cell[0], ReadNoise(mode="per_cell", sigma_cell_frac=0.0,
                           adc_bits=None, dac_bits=None), make_generator(seed, "z"),
                           backward_supported=False)  # fmt: skip
        return BoundedNoiseReader(inner, sg * wmax, make_generator(seed, f"bn_{tag}"), "sign")
    noise = ReadNoise(mode="per_cell", sigma_cell_frac=sg, adc_bits=None, dac_bits=None)
    return TileReader(lambda: healthy_cell[0], noise, make_generator(seed, f"gn_{tag}"),
                      backward_supported=False)  # fmt: skip


def _score(est, cost, vis, meta_r):
    reads = cost.mvm_fwd + cost.mvm_bwd
    return {"reads": reads, "r_cols": meta_r, "col_equiv": meta_r + max(reads - 1, 0),
            "line_f1": line_f1(est.rows, vis), "cell_f1": localization_f1(est.cells, vis),
            "guarantee": bool(est.meta.get("paper_guarantee_form", False))}  # fmt: skip


def run_eval(point: dict) -> list[dict]:
    torch.set_num_threads(1)
    tile, healthy = B._tile_state(point)
    bad, vis = B._inject(point, healthy, tile)
    wmax = float(tile.w.abs().max())
    seed, k = B._seed(point) + 7919 * point["k"], point["k"]
    dc = 1 if k <= 4 else 2 if k <= 32 else 3
    ds = {(a, d, c): b for a, d, c, b in B.designs(tile.w, k, point["fa"], seed)}
    rows = []
    for nm in ("bounded", "gauss"):
        for sg in point["sigmas"]:
            row = {"noise": nm, "sigma_c": sg, "n_visible": len(vis)}
            for arm, (stage, thr, rep) in ARMS.items():
                cur = [healthy]
                rd = _reader(nm, cur, wmax, sg, seed, f"{nm}_{arm}")
                if rep > 1:
                    rd = _AvgReader(rd, rep)
                mult = 3.0 if (nm == "gauss" and thr == "worst") else 1.0
                bu = mult * sg * wmax / rep**0.5
                loc = WeiRothPaperLocalizer(tile.w, bound_unit=bu, d_lines=D_LINES[k],
                                            d_cells=dc, cell_stage=stage, threshold=thr,
                                            seed=seed, fa=point["fa"])  # fmt: skip
                cur[0] = bad
                est, cost = loc.localize(rd)
                for key, v in _score(est, cost, vis, loc.H.shape[0]).items():
                    row[f"{key}_{arm}"] = v
            s = point["sel"][f"{k}|{sg}"]["csf"]
            cur = [healthy]
            rd = _reader(nm, cur, wmax, sg, seed, f"{nm}_csf")
            loc = ds[(s["algo"], s["dir"], s["cfg"])]()
            loc.calibrate(rd)
            cur[0] = bad
            est, cost = loc.localize(rd)
            for key, v in _score(est, cost, vis, 0).items():
                row[f"{key}_csf"] = v
            rows.append(row)
    return rows


def summarize(ev: pd.DataFrame, h8: pd.DataFrame) -> pd.DataFrame:
    h8c = KEYS + ["reads_roth", "f1_roth", "reads_csb", "f1_csb"]
    h8 = h8[h8c].rename(columns={c: f"{c}_h8" for c in h8c if c not in KEYS})
    m = ev.merge(h8, on=KEYS, how="left")
    rows = []
    for (nm, k, sg), g in m.groupby(["noise", "k", "sigma_c"]):
        r = {"noise": nm, "k": int(k), "sigma_c": float(sg), "n_instances": len(g)}
        for arm in [*ARMS, "csf"]:
            for q in ("reads", "r_cols", "col_equiv", "line_f1", "cell_f1"):
                r[f"{q}_{arm}"] = float(g[f"{q}_{arm}"].mean())
        for arm in ARMS:
            r[f"{arm}_line_reaches"] = int(r[f"line_f1_{arm}"] >= 0.9)
        r["csf_reaches"] = int(r["cell_f1_csf"] >= 0.9)
        for arm in CELL_ARMS:
            _lo, hi, ratio = paired_ratio_ci(g["reads_csf"].to_numpy(float),
                                            g[f"reads_{arm}"].to_numpy(float), n_boot=4000)  # fmt: skip
            r[f"ratio_csf_{arm}"], r[f"ratio_csf_{arm}_ci_upper"] = ratio, hi
            reach = r[f"cell_f1_{arm}"] >= 0.9
            r[f"row_ok_2x_{arm}"] = int(r["csf_reaches"] == 1 and ((not reach) or (
                ratio <= 0.5 and hi <= 0.6)))  # fmt: skip
        # best-of-two wr_cell arms: the baseline gets its better threshold
        r["wr_cell_any_reaches"] = int(any(r[f"cell_f1_{a}"] >= 0.9 for a in CELL_ARMS))
        ok = [a for a in CELL_ARMS if r[f"cell_f1_{a}"] >= 0.9]
        best = (min(ok, key=lambda a: r[f"reads_{a}"]) if ok
                else max(CELL_ARMS, key=lambda a: r[f"cell_f1_{a}"]))  # fmt: skip
        r["wr_beats_csf"] = int(
            bool(ok) and r[f"ratio_csf_{best}"] >= 1.0
        )  # a reaching W-R arm is no dearer
        r["row_ok_2x_wr_best"], r["ratio_csf_wr_best"] = (
            r[f"row_ok_2x_{best}"],
            r[f"ratio_csf_{best}"],
        )
        r["wr_line_any_reaches"] = int(max(r[f"{a}_line_reaches"] for a in ARMS if "line" in a))
        r["wr_line_1read_reaches"] = int(max(r["wr_line_worst_line_reaches"],
                                             r["wr_line_rss_line_reaches"]))  # fmt: skip
        r["h8_reads_roth"] = (
            float(g["reads_roth_h8"].mean()) if g["reads_roth_h8"].notna().all() else np.nan
        )
        r["h8_f1_roth"] = float(g["f1_roth_h8"].mean()) if g["f1_roth_h8"].notna().all() else np.nan
        rows.append(r)
    return pd.DataFrame(rows).sort_values(["noise", "sigma_c", "k"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--allow-dirty", action="store_true")
    ap.add_argument("--jobs", type=int, default=3)
    ap.add_argument("--out", default=None)
    ap.add_argument("--h8-dir", default=str(H8_DIR))
    a = ap.parse_args()
    ev_seeds = range(EV_SEEDS[0], EV_SEEDS[0] + (2 if a.smoke else 20))
    ckpts = [42] if a.smoke else C.CKPTS
    ks = [1, 16] if a.smoke else KS
    tag = "smoke" if a.smoke else "full"
    out = Path(
        a.out
        or B.REPO / "results" / "spectral" / ("e5_weiroth_fair" + ("_smoke" if a.smoke else ""))
    )
    h8d = Path(a.h8_dir)
    sel = json.loads((h8d / "selected_configs_full.json").read_text())
    h8_csv = max(h8d.glob("2026*")) / "eval_instances.csv"
    h8 = pd.read_csv(h8_csv)
    ev_pts = C.points("eval", TILE, ev_seeds, ks, SIGMAS, ckpts)
    if a.smoke:
        ev_pts = ev_pts[: 2 * len(ks) * 2]
    for p in ev_pts:
        p["sel"] = sel
    print(f"evaluation {len(ev_pts)} points", flush=True)
    ev = run_grid(run_eval, ev_pts, jobs=a.jobs, out_dir=out / f"_cells_eval_{tag}")
    vm = summarize(ev, h8)
    pd.set_option("display.width", 250, "display.max_columns", 60)
    cols = [
        "noise",
        "sigma_c",
        "k",
        "n_instances",
        "line_f1_wr_line_worst",
        "line_f1_wr_line_rss",
        "line_f1_wr_line_rss_R",
        "r_cols_wr_line_rss",
        "cell_f1_wr_cell_worst",
        "reads_wr_cell_worst",
        "cell_f1_wr_cell_rss",
        "reads_wr_cell_rss",
        "cell_f1_wr_cell_rss_R",
        "reads_wr_cell_rss_R",
        "cell_f1_csf",
        "reads_csf",
        "ratio_csf_wr_best",
        "row_ok_2x_wr_best",
    ]
    print(vm[cols].round(3).to_string(index=False))
    cfg = {"ks": ks, "sigmas": SIGMAS, "tile": TILE, "eval_seeds": list(EV_SEEDS), "ckpts": ckpts,
           "d_lines": D_LINES, "arms": ARMS, "h8_eval_csv": str(h8_csv)}  # fmt: skip
    rd = write_run(out, cfg, 0, vm, {}, extra={"mode": tag}, allow_dirty=a.allow_dirty, run_id=None)
    vm.to_csv(rd / "verdict_metrics.csv", index=False)
    ev.to_csv(rd / "eval_instances.csv", index=False)
    print("wrote", rd)


if __name__ == "__main__":
    main()
