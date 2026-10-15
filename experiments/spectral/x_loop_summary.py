"""Closed loop X.1/X.2: build verdict_metrics.csv for criteria/x_loop.yaml from the run dirs.

One row per (regime, seed). Accuracies are final test accuracy (%) from events.json summary.
Dense oracle references (D, C'-matched, R1) are the existing S1 runs of the mitigation paper
(results/_live9/resnet9_real_s1_*); the seed-42 D-oracle rerun is a reproducibility check (report).

  python experiments/spectral/x_loop_summary.py [--root results/spectral/x_loop] [--live results/_live9]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

SEEDS = (42, 101, 102)
TAG = "adaptive_s0.02"


def _ev(path: Path) -> dict | None:
    f = path / "events.json"
    if not f.exists():
        return None
    d = json.loads(f.read_text())
    return d if d.get("complete") else None


def _final(d: dict | None) -> float:
    return float(d["summary"]["final_test_acc"]) if d else float("nan")


def _loc(d: dict | None) -> dict:
    """Lifetime localization stats from the per-wave `localize` fields."""
    if not d:
        return {}
    locs = [e["localize"] for e in d["events"] if "localize" in e]
    if not locs:
        return {}
    rec = [
        lc["found_new_visible"] / lc["truth_new_visible"] for lc in locs if lc["truth_new_visible"]
    ]
    srec = [lc["shadow_found_new_visible"] / lc["truth_new_visible"] for lc in locs
            if lc["truth_new_visible"]]  # fmt: skip
    reads, sreads = sum(lc["reads"] for lc in locs), sum(lc["shadow_reads"] for lc in locs)
    last = locs[-1]
    return {
        "recall_vis_min": min(rec) if rec else 1.0,
        "shadow_recall_vis_min": min(srec) if srec else 1.0,
        "reads_total": reads,
        "shadow_reads_total": sreads,
        "reads_ratio": reads / sreads if sreads else float("nan"),
        "fallback_frac": sum(lc["fallback_tiles"] for lc in locs) / sum(lc["tiles"] for lc in locs),
        "fp_final": last["fp"],
        "fn_visible_final": last["fn_visible"],
        "tp_final": last["tp"],
    }


def build(root: Path, live: Path) -> pd.DataFrame:
    rows = []
    for s in SEEDS:
        d_est = _ev(root / f"x_dense_d_x{TAG}_seed{s}")
        c_est = _ev(root / f"x_dense_c_living_matched_x{TAG}_seed{s}")
        r = {"regime": "dense", "seed": s, "d_est": _final(d_est), "c_est": _final(c_est),
             "d_oracle": _final(_ev(live / f"resnet9_real_s1_d_seed{s}")),
             "c_oracle": _final(_ev(live / f"resnet9_real_s1_c_living_matched_seed{s}")),
             "floor": _final(_ev(live / f"resnet9_real_s1_r1_seed{s}")),
             "burst_d_est": d_est["summary"]["total_burst_batches"] if d_est else float("nan"),
             **_loc(d_est)}  # fmt: skip
        rows.append(r)
        d_est = _ev(root / f"x_sparse_d_x{TAG}_seed{s}")
        r = {"regime": "sparse", "seed": s, "d_est": _final(d_est), "c_est": float("nan"),
             "d_oracle": _final(_ev(root / f"x_sparse_d_xoracle_seed{s}")), "c_oracle": float("nan"),
             "floor": _final(_ev(root / f"x_sparse_r0_xoracle_seed{s}")),
             "burst_d_est": d_est["summary"]["total_burst_batches"] if d_est else float("nan"),
             **_loc(d_est)}  # fmt: skip
        rows.append(r)
    df = pd.DataFrame(rows)
    df["gap_d"] = df.d_est - df.d_oracle
    df["d_est_minus_c_oracle"] = df.d_est - df.c_oracle
    df["d_est_minus_c_est"] = df.d_est - df.c_est
    df["closure"] = (df.d_est - df.floor) / (df.d_oracle - df.floor)
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results/spectral/x_loop")
    ap.add_argument("--live", default="results/_live9")
    a = ap.parse_args()
    root = Path(a.root)
    df = build(root, Path(a.live))
    pd.set_option("display.width", 250, "display.max_columns", 40)
    print(df.round(3).to_string(index=False))
    rep = _ev(root / "x_dense_d_xoracle_seed42")
    if rep:
        print(f"repro: D-oracle seed 42 rerun {_final(rep):.2f} vs _live9 "
              f"{df.loc[(df.regime == 'dense') & (df.seed == 42), 'd_oracle'].item():.2f}")  # fmt: skip
    df.to_csv(root / "verdict_metrics.csv", index=False)
    print("wrote", root / "verdict_metrics.csv")


if __name__ == "__main__":
    main()
