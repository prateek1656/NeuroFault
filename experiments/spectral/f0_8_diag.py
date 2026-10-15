# ruff: noqa: C408
"""F0.8 diagnosis (POST-HOC, report only; never changes the F0.8 verdict): is the tier-1 vs tier-2
gap of the final F0.8 run (results/spectral/f0_8/20261011T151445_d931ec73_s0, up to 5.7 pt, almost
all on B8 in BN-folded / hotspot cells) caused by CrossSim's 8-bit DAC / ADC quantization?

Re-evaluates the worst cells (BN-folded network, seeds 101/102, D2 and D3 hotspot, 1d and 1yr, arms
b8/m4) with the SAME matrices/biases as F0.8 (same streams, same code path) on tier 1 and on tier 2
at four (ADC bits, DAC bits) settings: (8, 8) = the F0.8 tier 2, (16, 16), (16, 8), (8, 16).
If tier 2 at (16, 16) matches tier 1 within 0.5 pt, the gap is converter quantization (a hardware
effect the twin does not model), not a twin bug.

  python experiments/spectral/f0_8_diag.py --jobs 2 --allow-dirty
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import a_bnfold as ab
import f0_8_bridge as fb
import pandas as pd
import torch
from a_common import BN_OF, INCUMBENTS, TileCtx, best_of, make_drift
from f0_common import load_config

from neurofault.spectral.bnfold import SEED_OFFSET
from neurofault.spectral.bridge import TierTwo
from neurofault.spectral.harness import run_grid
from neurofault.spectral.types import make_generator

BITS = [(8, 8), (16, 16), (16, 8), (8, 16)]  # (adc, dac)
TIMES = {"1d": 86400.0, "1yr": 31557600.0}
ARMS = ["b8", "m4"]


def diag_job(p: dict) -> list[dict]:
    torch.set_num_threads(1)
    cfg, seed, sc = p["cfg"], p["seed"], p["scenario"]
    bridge = fb._bridge(seed, "folded", cfg["n_eval"])
    t2s = {b: TierTwo(bridge.model, dict(BN_OF), seed, adc_bits=b[0], dac_bits=b[1]) for b in BITS}
    saved = ab._install("cifar10", seed, bridge)
    rseed = seed + SEED_OFFSET
    rows: list[dict] = []
    try:
        mats = bridge.layer_mats()
        shapes = {k: tuple(v.shape) for k, v in mats.items()}
        mod = make_drift(sc["model"], sc.get("field") or "row_gradient", sc["rho"], 1.0)
        runs = []
        for layer, r0, c0, m, n in ab._tiles(bridge):
            ctx = TileCtx(seed, layer, r0, c0, m, n, 256)
            real = mod.sample(ctx.stp, make_generator(rseed, f"drift:{layer}:{r0}:{c0}"))
            run = ab.BFRun(ctx.slim(), cfg["sigma"], rseed)
            run.real = real
            runs.append(((layer, r0, c0, m, n), run))
        for tlabel, t in TIMES.items():
            W = {a: {} for a in ARMS}
            B = {a: {} for a in ARMS}
            for (layer, r0, c0, m, n), run in runs:
                res = run.at(t)
                for a in ARMS:
                    r = res[best_of(res, INCUMBENTS) if a == "b8" else "m4"]
                    fb._scatter(W[a], layer, shapes[layer], r0, c0, m, n, r["W"])
                    bd = B[a].setdefault(layer, torch.zeros(shapes[layer][0]))
                    bd[r0 : r0 + m] += r["b"].float()
            for a in ARMS:
                row = dict(seed=seed, scenario=sc["name"], arm=a, time=tlabel,
                           acc_t1=bridge.acc(W[a], B[a]))  # fmt: skip
                for (adc, dac), t2 in t2s.items():
                    t2.set_state(W[a], B[a])
                    row[f"acc_t2_a{adc}_d{dac}"] = t2.acc(bridge.xe, bridge.ye)
                rows.append(row)
        return rows
    finally:
        ab._restore(saved)


def main() -> int:
    cfg = load_config(Path(__file__).parent / "configs" / "f0_8.yaml", False)
    jobs = int(sys.argv[sys.argv.index("--jobs") + 1]) if "--jobs" in sys.argv else 1
    out = Path("results/spectral/f0_8_diag")
    pts = [dict(cfg=cfg, seed=s, scenario=sc) for sc in cfg["scenarios"] for s in (101, 102)]
    df = run_grid(diag_job, pts, jobs=jobs, out_dir=out / "_cells")
    for c in [c for c in df.columns if c.startswith("acc_t2_")]:
        df["gap_" + c.removeprefix("acc_t2_")] = (df.acc_t1 - df[c]).round(2)
    pd.set_option("display.width", 250)
    print(df.round(2).to_string(index=False))
    df.to_csv(out / "diag.csv", index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
