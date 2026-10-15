"""Hardening H8 (criteria/b_vs_prior.yaml): registered confirmation of the joint compressed-sensing
localizer (A5) against the published localizers - HPE-style syndrome + binary search (A2) and the
Wei & Roth disjunct code (A2b) - on FRESH fault seeds, with every family's configuration frozen on a
separate calibration split (the exploratory b_e1 grid chose the best config on the evaluation
instances themselves).

Families (design catalogue = b_e1.designs, unmodified): HPE = A2 fwd, ROTH = A2b fwd, CSF = A5 fwd,
CSB = A5 bidir. Selection rule per (k, sigma) and family (as C1b): fewest calibration-mean reads among
configs with calibration-mean cell F1 >= 0.93; fallback = highest F1. Calibration = tile origin 1,
fault seeds 3000-3019; evaluation = tile origin 0, fault seeds 60-79 (never used before: C1 used 0-19,
C1b 20-39, E1 0-4). Selected configs are written to disk before evaluation starts.

    python experiments/spectral/b_vs_prior.py [--smoke] [--allow-dirty] [--jobs 3]
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
from neurofault.spectral.metrics import localization_f1, paired_ratio_ci
from neurofault.spectral.readout import TileReader
from neurofault.spectral.types import ReadNoise, make_generator

FAMS = {"hpe": ("A2", "fwd"), "roth": ("A2b", "fwd"), "csf": ("A5", "fwd"), "csb": ("A5", "bidir")}
KS = [1, 4, 16, 64]
SIGMAS = [0.01, 0.02, 0.04]
TILE = 256
TARGET_CAL = 0.93


def _fam_designs(w, k, fa, seed):
    keep = set(FAMS.values())
    return [d for d in B.designs(w, k, fa, seed) if (d[0], d[1]) in keep]


def _run(reader_key, build, cur, healthy, bad, noise, seed, bidir, vis):
    reader = TileReader(lambda: cur[0], noise, make_generator(seed, f"read_{reader_key}"),
                        backward_supported=bidir)  # fmt: skip
    loc = build()
    cur[0] = healthy
    loc.calibrate(reader)
    cur[0] = bad
    est, cost = loc.localize(reader)
    return cost.mvm_fwd + cost.mvm_bwd, localization_f1(est.cells, vis)


def run_cal(point: dict) -> list[dict]:
    torch.set_num_threads(1)
    tile, healthy = B._tile_state(point)
    bad, vis = B._inject(point, healthy, tile)
    cur, seed = [healthy], B._seed(point) + 7919 * point["k"]
    rows = []
    for sg in point["sigmas"]:
        noise = ReadNoise(mode="per_cell", sigma_cell_frac=sg, adc_bits=None, dac_bits=None)
        reads, f1s = [], []
        for a, d, c, build in _fam_designs(tile.w, point["k"], point["fa"], seed):
            r, f = _run(f"{a}_{d}_{c}", build, cur, healthy, bad, noise, seed, d == "bidir", vis)
            reads.append(r)
            f1s.append(round(f, 5))
        rows.append({"sigma_c": sg, "n_visible": len(vis), "reads": json.dumps(reads),
                     "f1": json.dumps(f1s)})  # fmt: skip
    return rows


def select(cal: pd.DataFrame) -> dict:
    sel = {}
    for (k, sg), g in cal.groupby(["k", "sigma_c"]):
        names = [(a, d, c) for a, d, c, _ in _fam_designs(torch.zeros(TILE, TILE), int(k), 0.1, 0)]
        R = np.array([json.loads(s) for s in g.reads], float).mean(0)
        F = np.array([json.loads(s) for s in g.f1], float).mean(0)
        out = {}
        for fam, (a0, d0) in FAMS.items():
            idx = np.array([i for i, n in enumerate(names) if (n[0], n[1]) == (a0, d0)])
            ok = idx[F[idx] >= TARGET_CAL]
            if len(ok):
                j = ok[np.argmin(R[ok])]
            else:
                cand = idx[F[idx] >= F[idx].max() - 1e-12]
                j = cand[np.argmin(R[cand])]
            out[fam] = {"algo": a0, "dir": d0, "cfg": names[j][2], "cal_reads": float(R[j]),
                        "cal_f1": float(F[j]), "cal_reached_target": bool(F[j] >= TARGET_CAL)}  # fmt: skip
        sel[f"{int(k)}|{sg}"] = out
    return sel


def run_eval(point: dict) -> list[dict]:
    torch.set_num_threads(1)
    tile, healthy = B._tile_state(point)
    bad, vis = B._inject(point, healthy, tile)
    cur, seed = [healthy], B._seed(point) + 7919 * point["k"]
    ds = {(a, d, c): b for a, d, c, b in _fam_designs(tile.w, point["k"], point["fa"], seed)}
    rows = []
    for sg in point["sigmas"]:
        noise = ReadNoise(mode="per_cell", sigma_cell_frac=sg, adc_bits=None, dac_bits=None)
        row = {"sigma_c": sg, "n_visible": len(vis)}
        for fam, s in point["sel"][f"{point['k']}|{sg}"].items():
            key = (s["algo"], s["dir"], s["cfg"])
            r, f = _run(f"{fam}_{key}", ds[key], cur, healthy, bad, noise, seed,
                        s["dir"] == "bidir", vis)  # fmt: skip
            row[f"reads_{fam}"], row[f"f1_{fam}"] = r, f
        rows.append(row)
    return rows


def _row_ok(f1_cs, f1_roth, ratio, hi, bar, bar_hi):
    if f1_cs < 0.9:
        return 0
    return int(f1_roth < 0.9 or (ratio <= bar and hi <= bar_hi))


def summarize(ev: pd.DataFrame, sel: dict) -> pd.DataFrame:
    rows = []
    for (k, sg), g in ev.groupby(["k", "sigma_c"]):
        r = {"k": int(k), "sigma_c": float(sg), "tile_size": TILE, "n_instances": len(g)}
        for fam in FAMS:
            r[f"reads_{fam}"] = float(g[f"reads_{fam}"].mean())
            r[f"f1_{fam}"] = float(g[f"f1_{fam}"].mean())
            r[f"cfg_{fam}"] = sel[f"{int(k)}|{sg}"][fam]["cfg"]
        for cs in ("csb", "csf"):
            _lo, hi, ratio = paired_ratio_ci(g[f"reads_{cs}"].to_numpy(float),
                                            g["reads_roth"].to_numpy(float), n_boot=4000)  # fmt: skip
            r[f"ratio_{cs}_roth"], r[f"ratio_{cs}_roth_ci_upper"] = ratio, hi
            r[f"speedup_{cs}_vs_roth"] = 1.0 / ratio
        r["roth_reaches"] = int(r["f1_roth"] >= 0.9)
        r["hpe_reaches"] = int(r["f1_hpe"] >= 0.9)
        r["row_ok_2x"] = _row_ok(r["f1_csb"], r["f1_roth"], r["ratio_csb_roth"],
                                 r["ratio_csb_roth_ci_upper"], 0.5, 0.6)  # fmt: skip
        r["row_ok_4x"] = _row_ok(r["f1_csb"], r["f1_roth"], r["ratio_csb_roth"],
                                 r["ratio_csb_roth_ci_upper"], 0.25, 0.33)  # fmt: skip
        r["row_ok_fwd_2x"] = _row_ok(r["f1_csf"], r["f1_roth"], r["ratio_csf_roth"],
                                     r["ratio_csf_roth_ci_upper"], 0.5, 0.6)  # fmt: skip
        rows.append(r)
    return pd.DataFrame(rows).sort_values(["sigma_c", "k"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--allow-dirty", action="store_true")
    ap.add_argument("--jobs", type=int, default=3)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    cal_seeds = range(3000, 3002) if a.smoke else range(3000, 3020)
    ev_seeds = range(60, 62) if a.smoke else range(60, 80)
    ckpts = [42] if a.smoke else C.CKPTS
    ks = [1, 16] if a.smoke else KS
    tag = "smoke" if a.smoke else "full"
    out = Path(
        a.out or B.REPO / "results" / "spectral" / ("b_vs_prior" + ("_smoke" if a.smoke else ""))
    )
    cal_pts = C.points("cal", TILE, cal_seeds, ks, SIGMAS, ckpts)
    ev_pts = C.points("eval", TILE, ev_seeds, ks, SIGMAS, ckpts)
    if a.smoke:
        cal_pts, ev_pts = cal_pts[: 2 * len(ks) * 2], ev_pts[: 2 * len(ks) * 2]
    print(f"calibration {len(cal_pts)} points, evaluation {len(ev_pts)} points", flush=True)
    cal = run_grid(run_cal, cal_pts, jobs=a.jobs, out_dir=out / f"_cells_cal_{tag}")
    sel = select(cal)
    out.mkdir(parents=True, exist_ok=True)
    sel_path = out / f"selected_configs_{tag}.json"
    sel_path.write_text(json.dumps(sel, indent=1, sort_keys=True))
    print("frozen", sel_path, flush=True)
    for p in ev_pts:
        p["sel"] = sel
    ev = run_grid(run_eval, ev_pts, jobs=a.jobs, out_dir=out / f"_cells_eval_{tag}")
    vm = summarize(ev, sel)
    pd.set_option("display.width", 250, "display.max_columns", 40)
    cols = ["sigma_c", "k", "n_instances"] + [f"{m}_{f}" for f in FAMS for m in ("reads", "f1")] + [
        "speedup_csb_vs_roth", "ratio_csb_roth_ci_upper", "speedup_csf_vs_roth", "row_ok_2x",
        "row_ok_4x", "row_ok_fwd_2x"]  # fmt: skip
    print(vm[cols].round(3).to_string(index=False))
    cfg = {"ks": ks, "sigmas": SIGMAS, "tile": TILE, "cal_seeds": [min(cal_seeds), max(cal_seeds)],
           "eval_seeds": [min(ev_seeds), max(ev_seeds)], "ckpts": ckpts, "cal_target": TARGET_CAL,
           "families": FAMS, "selected": sel}  # fmt: skip
    rd = write_run(out, cfg, 0, vm, {"target_f1": TARGET_CAL}, extra={"mode": tag},
                   allow_dirty=a.allow_dirty, run_id=None)  # fmt: skip
    vm.to_csv(rd / "verdict_metrics.csv", index=False)
    ev.to_csv(rd / "eval_instances.csv", index=False)
    print("wrote", rd)


if __name__ == "__main__":
    main()
