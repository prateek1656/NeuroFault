"""Direction B, E0 (baselines + noise calibration, k=1) and E1c (regime crossover density).

Arm P1.4. Thresholds the verdict reads are in criteria/b_claims.yaml
(frozen, never edited here): this script emits `probing_win_max_density` (E1c).

    python experiments/spectral/b_e0_e1c.py --exp e0|e1c [--smoke] [--allow-dirty] [--jobs 8]

Tiles: first full 256x256 tiles of layers in results/_live2/resnet9_predeploy_seed*/pre_deploy.pt.
Every localizer touches the tile only through a counted TileReader; w_ref (the digital programmed
weights) is the stored reference. Truth for scoring = faults with |delta_w| >= visible_frac*max|w|
(f1_all additionally scores against every injected cell).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
import yaml

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from neurofault.spectral.encoding import encode
from neurofault.spectral.faults import inject
from neurofault.spectral.harness import run_grid, write_run
from neurofault.spectral.localize import (
    FullReadoutLocalizer,
    HPELocalizer,
    ResidualLocalizer,
    RothLocalizer,
    TwoStageLocalizer,
)
from neurofault.spectral.metrics import localization_f1
from neurofault.spectral.readout import TileReader
from neurofault.spectral.tiles import load_checkpoint
from neurofault.spectral.types import (
    EncodingSpec,
    FaultSpec,
    ReadNoise,
    Tile,
    TileSpec,
    make_generator,
)

CFG_DIR = Path(__file__).resolve().parent / "configs"


@lru_cache(maxsize=8)
def _ckpt(seed: int) -> dict:
    return load_checkpoint(seed)


# ------------------------------------------------------------------ algorithm sets


def e0_algos(w, reps, fa, seed):
    """(label, family, budget, localizer) for k=1 (design d = 1)."""
    out = [("A1", "A1", 0, FullReadoutLocalizer(w, fa=fa, seed=seed))]
    for r in reps:
        out.append(
            (
                "A2_argmax",
                "A2",
                r,
                HPELocalizer(w, n_dense=r, n_bisect=max(1, r // 2), fa=fa, seed=seed),
            )
        )
        out.append(("A3", "A3", r, ResidualLocalizer(w, n_fwd=r, n_bwd=r, fa=fa, seed=seed)))
        out.append(("A4", "A4", r, TwoStageLocalizer(w, n_fwd=r, n_bwd=r, d2=1, fa=fa, seed=seed)))
        out.append(
            (
                "A2b_syndrome",
                "A2b",
                r,
                RothLocalizer(w, d_lines=1, d_cells=1, n_dense=r, fa=fa, seed=seed),
            )
        )
        out.append(
            (
                "A2b_direct",
                "A2b",
                r,
                RothLocalizer(
                    w, d_lines=1, d_cells=1, n_dense=r, stage1="direct", fa=fa, seed=seed
                ),
            )
        )
    for r in (reps[len(reps) // 2], reps[-1]):
        out.append(
            (
                "A2_syndrome",
                "A2",
                r,
                HPELocalizer(
                    w, n_dense=r, n_bisect=max(1, r // 2), decoder="syndrome", fa=fa, seed=seed
                ),
            )
        )
    return out


def e1c_algos(w, density, fa, seed):
    m, n = w.shape
    s = density * n
    d = max(1, math.ceil(s + 3 * math.sqrt(s)))
    out = [("A1", "A1", 0, FullReadoutLocalizer(w, fa=fa, seed=seed))]
    for npr in (16, 64):
        out.append(
            (
                f"A3_prune{npr}",
                "A3",
                npr,
                ResidualLocalizer(w, n_fwd=4, n_bwd=4, n_prune=npr, fa=fa, seed=seed),
            )
        )
    out.append(("A4", "A4", d, TwoStageLocalizer(w, n_fwd=4, n_bwd=4, d2=d, fa=fa, seed=seed)))
    out.append(
        (
            "A2b",
            "A2b",
            d,
            RothLocalizer(w, d_lines=m, d_cells=d, n_dense=4, stage1="direct", fa=fa, seed=seed),
        )
    )
    return out


# ------------------------------------------------------------------ worker


def _tile_state(point):
    spec = TileSpec(
        point["ckpt_seed"],
        point["layer"],
        point["r0"],
        point["c0"],
        point["tile_size"],
        point["tile_size"],
        point["tile_size"],
    )
    W = _ckpt(point["ckpt_seed"])[point["layer"]]
    t = point["tile_size"]
    tile = Tile(spec, W[point["r0"] : point["r0"] + t, point["c0"] : point["c0"] + t].clone())
    enc = EncodingSpec(**point["encoding"])
    return tile, encode(tile, enc)


def run_point(point: dict) -> list[dict]:
    torch.set_num_threads(1)
    tile, healthy = _tile_state(point)
    m, n = tile.w.shape
    fs = dict(point["fault"])
    if "density" in point:
        fs["density"] = point["density"]
    gen = make_generator(
        point["fault_seed"] * 1000 + point["ckpt_seed"] + point["r0"] + point["c0"], "faults"
    )
    bad, truth = inject(healthy, FaultSpec(**fs), gen)
    wmax = float(tile.w.abs().max())
    vis = truth.cells[truth.delta_w.abs() >= point["visible_frac"] * wmax]
    rows = []
    cur = [healthy]
    for sigma in point["sigmas"]:
        noise = ReadNoise(mode="per_cell", sigma_cell_frac=sigma, adc_bits=None, dac_bits=None)
        seed = (
            point["fault_seed"]
            + 100003 * point["ckpt_seed"]
            + 1009 * point["r0"]
            + 31 * point["c0"]
        )
        if point["experiment"] == "e0":
            algos = e0_algos(tile.w, point["reps"], point["fa"], seed)
        else:
            algos = e1c_algos(tile.w, point["density"], point["fa"], seed)
        for label, fam, budget, loc in algos:
            cur[0] = healthy
            reader = TileReader(
                lambda: cur[0], noise, make_generator(seed, f"read_{label}_{budget}")
            )
            cal = loc.calibrate(reader)
            cur[0] = bad
            est, cost = loc.localize(reader)
            f1 = localization_f1(est.cells, vis)
            f1_all = localization_f1(est.cells, truth.cells)
            hit = _hits(est.cells, vis, n)
            reads = cost.mvm_fwd + cost.mvm_bwd
            rows.append(
                {
                    "sigma_c": sigma,
                    "algo": label,
                    "family": fam,
                    "budget": budget,
                    "reads": reads,
                    "reads_fwd": cost.mvm_fwd,
                    "reads_bwd": cost.mvm_bwd,
                    "calib_reads": cal.mvm_fwd + cal.mvm_bwd,
                    "reads_over_n": reads / n,
                    "f1": f1,
                    "f1_all": f1_all,
                    "success": float(f1 == 1.0),
                    "precision": hit / max(len(est.cells), 1)
                    if len(est.cells)
                    else float(len(vis) == 0),
                    "recall": hit / len(vis) if len(vis) else 1.0,
                    "n_pred": len(est.cells),
                    "n_visible": len(vis),
                    "n_truth": len(truth.cells),
                    "tile_size": m,
                }
            )
    return rows


def _hits(pred, truth, n):
    if len(pred) == 0 or len(truth) == 0:
        return 0
    a = set((pred[:, 0] * n + pred[:, 1]).tolist())
    b = set((truth[:, 0] * n + truth[:, 1]).tolist())
    return len(a & b)


# ------------------------------------------------------------------ driver


def build_points(cfg):
    pts = []
    t = cfg["tile_size"]
    for cs in cfg["ckpt_seeds"]:
        for layer in cfg["layers"]:
            M, N = load_checkpoint(cs)[layer].shape
            origins = [(r, c) for r in range(0, M - t + 1, t) for c in range(0, N - t + 1, t)]
            for r0, c0 in origins[: cfg["tiles_per_layer"]]:
                for fseed in range(cfg["fault_seeds"]):
                    base = {
                        "experiment": cfg["experiment"],
                        "ckpt_seed": cs,
                        "layer": layer,
                        "r0": r0,
                        "c0": c0,
                        "tile_size": t,
                        "fault_seed": fseed,
                        "sigmas": cfg["sigmas"],
                        "encoding": cfg["encoding"],
                        "fault": cfg["fault"],
                        "visible_frac": cfg["visible_frac"],
                        "fa": cfg["fa"],
                    }
                    if cfg["experiment"] == "e0":
                        pts.append({**base, "reps": cfg["reps"]})
                    else:
                        for dens in cfg["densities"]:
                            pts.append({**base, "density": dens})
    return pts


def summarize_e0(df):
    g = (
        df.groupby(["sigma_c", "algo", "budget"])
        .agg(
            reads=("reads", "median"),
            success=("success", "mean"),
            f1=("f1", "mean"),
            n=("success", "size"),
        )
        .reset_index()
    )
    best = (
        g[g.success >= 0.9]
        .sort_values("reads")
        .groupby(["sigma_c", "algo"])
        .head(1)[["sigma_c", "algo", "budget", "reads", "success"]]
    )
    return g, best


def summarize_e1c(df, cfg):
    g = (
        df.groupby(["sigma_c", "density", "algo"])
        .agg(
            reads=("reads", "median"),
            reads_over_n=("reads_over_n", "median"),
            f1=("f1", "median"),
            f1_mean=("f1", "mean"),
            precision=("precision", "mean"),
            recall=("recall", "mean"),
        )
        .reset_index()
    )
    wins = {}
    for (s, d), sub in g.groupby(["sigma_c", "density"]):
        a1 = sub[sub.algo == "A1"].reads.iloc[0]
        pr = sub[(sub.algo != "A1") & (sub.f1 >= cfg["win_f1"]) & (sub.reads < a1)]
        wins[(s, d)] = bool(len(pr))
    return g, wins


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", choices=["e0", "e1c"], required=True)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--allow-dirty", action="store_true")
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    cfg = yaml.safe_load((CFG_DIR / f"b_{a.exp}.yaml").read_text())
    smoke = cfg.pop("smoke")
    if a.smoke:
        cfg.update(smoke)
    tag = "smoke" if a.smoke else "full"
    out = Path(a.out or REPO / "results" / "spectral" / f"b_{a.exp}")
    pts = build_points(cfg)
    print(f"{a.exp} {tag}: {len(pts)} points", flush=True)
    df = run_grid(run_point, pts, jobs=a.jobs, out_dir=out / f"_cells_{tag}")
    extra = {"mode": tag}
    if a.exp == "e0":
        g, best = summarize_e0(df)
        print(best.to_string(index=False))
        extra["reads_at_90pct_success"] = best.to_dict("records")
        df["density"] = np.nan
    else:
        df["density"] = df["density"].astype(float)
        g, wins = summarize_e1c(df, cfg)
        primary = (
            cfg["primary_sigma"] if cfg["primary_sigma"] in cfg["sigmas"] else cfg["sigmas"][0]
        )
        win_ds = [d for (s, d), w in wins.items() if s == primary and w]
        pmax = max(win_ds) if win_ds else 0.0
        df["probing_win_max_density"] = np.where(df.sigma_c == primary, pmax, np.nan)
        print(g.to_string(index=False))
        print("wins:", {f"{s}:{d}": w for (s, d), w in wins.items()})
        print("probing_win_max_density (sigma", primary, "):", pmax)
        extra.update(
            probing_win_max_density=pmax, wins={f"{s}:{d}": w for (s, d), w in wins.items()}
        )
    rd = write_run(
        out,
        cfg,
        0,
        df,
        params={"visible_frac": cfg["visible_frac"], "fa": cfg["fa"]},
        extra=extra,
        allow_dirty=a.allow_dirty,
        run_id=None,
    )
    g.to_csv(rd / "summary.csv", index=False)
    (rd / "summary.json").write_text(json.dumps(extra, indent=2, default=str))
    print("wrote", rd)


if __name__ == "__main__":
    main()
