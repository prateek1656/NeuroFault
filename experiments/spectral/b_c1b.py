"""Direction B, registered follow-up C1b (criteria/b_c1b.yaml).

Same machinery as b_c1_prereg.py (imported, unmodified) with the registered changes:
margin selection (calibration mean F1 >= 0.93, fallback highest F1), calibration = tile index 1 and
fault seeds 1000-1019, evaluation = tile index 0 and FRESH fault seeds 20-39. Selected configs are
written to disk before evaluation starts.

    python experiments/spectral/b_c1b.py [--smoke] [--allow-dirty] [--jobs 3]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import b_c1_prereg as C
import b_e1 as B

from neurofault.spectral.harness import run_grid, write_run
from neurofault.spectral.metrics import paired_ratio_ci

TARGET = 0.93


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--allow-dirty", action="store_true")
    ap.add_argument("--jobs", type=int, default=3)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    cal_seeds = range(1000, 1002) if a.smoke else range(1000, 1020)
    ev_seeds = range(20, 23) if a.smoke else range(20, 40)
    tiles = [256] if a.smoke else C.TILES
    ckpts = [42] if a.smoke else C.CKPTS
    out = Path(a.out or B.REPO / "results" / "spectral" / "b_c1b")
    tag = "smoke" if a.smoke else "full"
    cal_pts, ev_pts = [], []
    for t in tiles:
        cal_pts += C.points("cal", t, cal_seeds, C.KS, C.SIGMAS, ckpts)
        ev_pts += C.points("eval", t, ev_seeds, C.KS, C.SIGMAS, ckpts)
    print(f"calibration {len(cal_pts)} points, evaluation {len(ev_pts)} points", flush=True)
    cal = run_grid(B.run_e1, cal_pts, jobs=a.jobs, out_dir=out / f"_cells_cal_{tag}")
    sel = C.select(cal, target=TARGET)
    out.mkdir(parents=True, exist_ok=True)
    sel_path = out / f"selected_configs_{tag}.json"
    sel_path.write_text(json.dumps(sel, indent=1, sort_keys=True))
    print("frozen", sel_path, flush=True)
    for p in ev_pts:
        p["sel"] = sel
    ev = run_grid(C.run_eval, ev_pts, jobs=a.jobs, out_dir=out / f"_cells_eval_{tag}")
    rows = []
    for (k, sg, tsz), g in ev.groupby(["k", "sigma_c", "tile_size"]):
        rf, rb = g.reads_fwd.to_numpy(float), g.reads_bidir.to_numpy(float)
        lo, hi, ratio = paired_ratio_ci(rb, rf, n_boot=4000)
        f1b = float(g.f1_bidir.mean())
        s = sel[f"{tsz}|{k}|{sg}"]
        rows.append(
            {
                "tile_size": int(tsz),
                "k": int(k),
                "sigma_c": float(sg),
                "n_instances": len(g),
                "fwd_cfg": f"{s['fwd']['algo']}:{s['fwd']['cfg']}",
                "bidir_cfg": f"{s['bidir']['algo']}:{s['bidir']['cfg']}",
                "cal_f1_fwd": s["fwd"]["cal_f1"],
                "cal_f1_bidir": s["bidir"]["cal_f1"],
                "cal_reached_fwd": int(s["fwd"]["cal_reached_target"]),
                "cal_reached_bidir": int(s["bidir"]["cal_reached_target"]),
                "cell_f1_bidir": f1b,
                "cell_f1_fwd": float(g.f1_fwd.mean()),
                "reads_fwd": float(rf.mean()),
                "reads_bidir": float(rb.mean()),
                "reads_ratio": ratio,
                "reads_ratio_ci_lo": lo,
                "reads_ratio_ci_upper": hi,
                "c1b_row_ok": int(f1b >= 0.9 and ratio <= 0.7 and hi <= 0.8),
            }
        )
    vm = pd.DataFrame(rows).sort_values(["tile_size", "k", "sigma_c"])
    pd.set_option("display.width", 250, "display.max_columns", 40)
    print(vm.to_string(index=False))
    cfg = {"ks": C.KS, "sigmas": C.SIGMAS, "tiles": tiles, "cal_seeds": [min(cal_seeds), max(cal_seeds)],
           "eval_seeds": [min(ev_seeds), max(ev_seeds)], "ckpts": ckpts, "cal_target": TARGET,
           "selected": sel}  # fmt: skip
    rd = write_run(out, cfg, 0, vm, {"target_f1": TARGET}, extra={"mode": tag},
                   allow_dirty=a.allow_dirty, run_id=None)  # fmt: skip
    vm.to_csv(rd / "verdict_metrics.csv", index=False)
    ev.to_csv(rd / "eval_instances.csv", index=False)
    print("wrote", rd)


if __name__ == "__main__":
    main()
