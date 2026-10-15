"""Direction B, pre-registered C1 run (C1): calibration split -> frozen configs -> evaluation.

C1: 3 ckpts x ALL conv layers (whose weights hold at least one full tile) x 20 fault seeds, sigma_c in
{1%, 2%}, k in {1,2,4,8,16}; best bidirectional vs best forward-only design, the forward-only side with
the same tuning effort. Selection is done on a CALIBRATION split that shares no fault seed and no
tile with the evaluation split:
  calibration: tile index 1 of each layer, fault seeds 1000..1000+n_cal-1
  evaluation : tile index 0 of each layer, fault seeds 0..19
For every (tile_size, k, sigma) and each family (fwd / bidir) the configuration with the fewest mean
reads among those whose calibration mean F1 >= 0.9 is frozen (if none reaches 0.9: max mean F1,
fewest reads on ties) and written to selected_configs.json BEFORE the evaluation starts. Evaluation
runs only those two configurations on identical fault maps; the reads ratio and its paired
bootstrap CI are computed over evaluation instances.

    python experiments/spectral/b_c1_prereg.py [--smoke] [--allow-dirty] [--jobs 4]
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
import b_e1 as B

REPO = B.REPO
from neurofault.spectral.harness import run_grid, write_run
from neurofault.spectral.metrics import localization_f1, paired_ratio_ci
from neurofault.spectral.readout import TileReader
from neurofault.spectral.tiles import load_checkpoint
from neurofault.spectral.types import ReadNoise, make_generator

KS = [1, 2, 4, 8, 16]
SIGMAS = [0.01, 0.02]
TILES = [256, 128, 512]
ENC = {"kind": "offset_refcol", "g_min_frac": 0.01}
FAULT = {"kind": "stuck_random", "rail": "mixed", "device": "pos"}
CKPTS = [42, 101, 102]
CONV = [
    "layer1.0.0.weight",
    "res1.conv_a.weight",
    "res1.conv_b.weight",
    "layer2.0.0.weight",
    "layer3.0.0.weight",
    "res2.conv_a.weight",
    "res2.conv_b.weight",
]


def layers_for(t):
    shp = load_checkpoint(42)
    return [L for L in CONV if min(shp[L].shape) >= t]


def points(split, tsz, seeds, ks, sigmas, ckpts):
    t, pts = tsz, []
    for cs in ckpts:
        W = load_checkpoint(cs)
        for L in layers_for(t):
            M, N = W[L].shape
            origins = [(r, c) for r in range(0, M - t + 1, t) for c in range(0, N - t + 1, t)]
            r0, c0 = origins[0 if split == "eval" else 1]
            for fs in seeds:
                for k in ks:
                    pts.append(
                        {
                            "experiment": f"c1_{split}",
                            "ckpt_seed": cs,
                            "layer": L,
                            "r0": r0,
                            "c0": c0,
                            "tile_size": t,
                            "fault_seed": fs,
                            "k": k,
                            "sigmas": sigmas,
                            "encoding": ENC,
                            "fault": FAULT,
                            "visible_frac": 0.1,
                            "fa": 0.1,
                        }
                    )
    return pts


def select(raw, target=0.9):
    """{(tile_size,k,sigma): {fwd: [algo,dir,cfg], bidir: [...]}} from calibration rows."""
    sel, cat = {}, {}
    for (k, sg, tsz), g in raw.groupby(["k", "sigma_c", "tile_size"]):
        names = cat.setdefault(
            (k, tsz), [(a, d, c) for a, d, c, _ in B.designs(torch.zeros(tsz, tsz), int(k), 0.1, 0)]
        )
        R = np.array([json.loads(s) for s in g.reads], float).mean(0)
        F = np.array([json.loads(s) for s in g.f1], float).mean(0)
        out = {}
        for d in ("fwd", "bidir"):
            idx = np.array([i for i, n in enumerate(names) if n[1] == d])
            ok = idx[F[idx] >= target]
            if len(ok):
                j = ok[np.argmin(R[ok])]
            else:
                best = F[idx].max()
                cand = idx[F[idx] >= best - 1e-12]
                j = cand[np.argmin(R[cand])]
            out[d] = {
                "algo": names[j][0],
                "cfg": names[j][2],
                "cal_reads": float(R[j]),
                "cal_f1": float(F[j]),
                "cal_reached_target": bool(F[j] >= target),
                "cal_instances": len(g),
            }
        sel[f"{tsz}|{k}|{sg}"] = out
    return sel


def run_eval(point):
    torch.set_num_threads(1)
    tile, healthy = B._tile_state(point)
    bad, vis = B._inject(point, healthy, tile)
    cur = [healthy]
    seed = B._seed(point) + 7919 * point["k"]
    ds = {(a, d, c): b for a, d, c, b in B.designs(tile.w, point["k"], point["fa"], seed)}
    rows = []
    for sg in point["sigmas"]:
        noise = ReadNoise(mode="per_cell", sigma_cell_frac=sg, adc_bits=None, dac_bits=None)
        row = {"sigma_c": sg, "n_visible": len(vis)}
        for d in ("fwd", "bidir"):
            s = point["sel"][f"{point['tile_size']}|{point['k']}|{sg}"][d]
            key = (s["algo"], d, s["cfg"])
            reader = TileReader(
                lambda: cur[0],
                noise,
                make_generator(seed, f"read_{key}"),
                backward_supported=d == "bidir",
            )
            loc = ds[key]()
            cur[0] = healthy
            loc.calibrate(reader)
            cur[0] = bad
            est, cost = loc.localize(reader)
            row[f"reads_{d}"] = cost.mvm_fwd + cost.mvm_bwd
            row[f"f1_{d}"] = localization_f1(est.cells, vis)
        rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--allow-dirty", action="store_true")
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    n_cal, n_eval = (2, 3) if a.smoke else (5, 20)
    tiles = [256] if a.smoke else TILES
    ckpts = [42] if a.smoke else CKPTS
    out = Path(a.out or REPO / "results" / "spectral" / "b_c1_prereg")
    tag = "smoke" if a.smoke else "full"
    cal_pts, ev_pts = [], []
    for t in tiles:
        cal_pts += points("cal", t, range(1000, 1000 + n_cal), KS, SIGMAS, ckpts)
        ev_pts += points("eval", t, range(n_eval), KS, SIGMAS, ckpts)
    print(f"calibration {len(cal_pts)} points, evaluation {len(ev_pts)} points", flush=True)
    cal = run_grid(B.run_e1, cal_pts, jobs=a.jobs, out_dir=out / f"_cells_cal_{tag}")
    sel = select(cal)
    out.mkdir(parents=True, exist_ok=True)
    sel_path = out / f"selected_configs_{tag}.json"
    sel_path.write_text(json.dumps(sel, indent=1, sort_keys=True))
    print("frozen", sel_path, flush=True)
    for p in ev_pts:
        p["sel"] = sel
    ev = run_grid(run_eval, ev_pts, jobs=a.jobs, out_dir=out / f"_cells_eval_{tag}")
    rows = []
    for (k, sg, tsz), g in ev.groupby(["k", "sigma_c", "tile_size"]):
        rf, rb = g.reads_fwd.to_numpy(float), g.reads_bidir.to_numpy(float)
        lo, hi, ratio = paired_ratio_ci(rb, rf, n_boot=4000)
        f1b = float(g.f1_bidir.mean())
        s = sel[f"{tsz}|{k}|{sg}"]
        rows.append(
            {
                "k": int(k),
                "sigma_c": float(sg),
                "tile_size": int(tsz),
                "n_instances": len(g),
                "fwd_cfg": f"{s['fwd']['algo']}:{s['fwd']['cfg']}",
                "bidir_cfg": f"{s['bidir']['algo']}:{s['bidir']['cfg']}",
                "cal_f1_fwd": s["fwd"]["cal_f1"],
                "cal_f1_bidir": s["bidir"]["cal_f1"],
                "reads_fwd": float(rf.mean()),
                "reads_bidir": float(rb.mean()),
                "cell_f1_fwd": float(g.f1_fwd.mean()),
                "cell_f1_bidir": f1b,
                "reads_ratio": ratio,
                "reads_ratio_ci_lo": lo,
                "reads_ratio_ci_upper": hi,
                "c1_row_ok": int(f1b >= 0.9 and ratio <= 0.7 and hi <= 0.8),
            }
        )
    vm = pd.DataFrame(rows).sort_values(["tile_size", "k", "sigma_c"])
    pd.set_option("display.width", 250, "display.max_columns", 40)
    print(vm.to_string(index=False))
    cfg = {"ks": KS, "sigmas": SIGMAS, "tiles": tiles, "n_cal_seeds": n_cal, "n_eval_seeds": n_eval,
           "ckpts": ckpts, "selected": sel}  # fmt: skip
    rd = write_run(out, cfg, 0, vm, {"target_f1": 0.9}, extra={"mode": tag},
                   allow_dirty=a.allow_dirty, run_id=None)  # fmt: skip
    vm.to_csv(rd / "verdict_metrics.csv", index=False)
    ev.to_csv(rd / "eval_instances.csv", index=False)
    print("wrote", rd)


if __name__ == "__main__":
    main()
